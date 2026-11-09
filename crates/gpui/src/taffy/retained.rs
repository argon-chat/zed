// Copyright 2022 - 2025 Zed Industries, Inc.
// Copyright 2026 Longbridge.
// SPDX-License-Identifier: Apache-2.0
// Measurement replay and final-width settling adapted from longbridge/gpui-fast,
// commit fb8cf9413777e43e7c38bf5917647ee984453f37, crates/gpui/src/fast/layout.rs.
// Modified for Taffy 0.13, complete style comparisons, per-window opt-in, scoped
// identity, bounded logs, fresh closures, and DPI/removal invalidation.

//! Optional per-window Taffy retention. Measurement replay/settling follows
//! gpui-fast (pinned fb8cf); policy and identity are local to this module.

use super::{AvailableSpace, EXPECT_MESSAGE, LayoutId, MeasureFn, NodeContext, TaffyLayoutEngine};
use crate::{App, GlobalElementId, Pixels, Size, Window};
use collections::FxHashMap;
use std::{cell::RefCell, mem, rc::Rc};
use taffy::{TraversePartialTree as _, tree::NodeId};

/// Work retained layout performed during an opt-in profiling capture.
#[derive(Clone, Copy, Debug, Default)]
pub struct VnRetainedLayoutStats {
    /// New Taffy nodes allocated during this capture.
    pub nodes_created: u64,
    /// Requests served by a node retained from a previous frame.
    pub nodes_reused: u64,
    /// Changed converted styles written to Taffy.
    pub style_writes: u64,
    /// Changed child lists written to Taffy.
    pub child_writes: u64,
    /// Calls to current measurement closures, including replay and settling.
    pub probe_calls: u64,
    /// Measured nodes whose recorded probes all matched the new closure.
    pub probe_replays: u64,
    /// Measured nodes invalidated because a recorded probe changed size.
    pub probe_mismatches: u64,
    /// Measurements taken again at their final laid out content width.
    pub settle_calls: u64,
    /// Transient or unclaimed Taffy nodes removed at frame end.
    pub nodes_freed: u64,
}

// A token lives only between one request's claim and remember calls. Buckets
// are recycled at frame end, after every token has been consumed.
struct Slot {
    bucket: usize,
    index: usize,
}

struct Entry {
    id: LayoutId,
    log: Option<Rc<RefCell<MeasureLog>>>,
}

#[derive(Default)]
struct ScopeBucket {
    claimed_in_frame: u64,
    next_slot: usize,
    entries: Vec<Option<Entry>>,
}

#[derive(Default)]
pub(super) struct Retention {
    pub(super) enabled: bool,
    pub(super) stats: Option<VnRetainedLayoutStats>,
    frame: u64,
    scale: Option<f32>,
    // Full GlobalElementId equality still checks the complete path, including
    // on a cached-hash collision. Only the request-local token is numeric.
    scopes: FxHashMap<GlobalElementId, usize>,
    buckets: Vec<Option<ScopeBucket>>,
    free_buckets: Vec<usize>,
    retained_nodes: usize,
    claimed_nodes: usize,
    transient: Vec<LayoutId>,
    unstretched: FxHashMap<LayoutId, taffy::style::Style>,
    pending: Vec<(LayoutId, Rc<RefCell<MeasureLog>>)>,
    measured: FxHashMap<NodeId, Option<Pixels>>,
}

type Probe = (Size<Option<Pixels>>, Size<AvailableSpace>, Size<Pixels>);

#[derive(Default)]
struct MeasureLog {
    probes: Vec<Probe>,
    overflowed: bool,
}

impl MeasureLog {
    fn record(
        &mut self,
        known: Size<Option<Pixels>>,
        available: Size<AvailableSpace>,
        size: Size<Pixels>,
    ) {
        if self.overflowed {
            return;
        }
        if let Some(index) = self
            .probes
            .iter()
            .position(|(k, a, _)| *k == known && *a == available)
        {
            self.probes.remove(index);
        } else if self.probes.len() == 16 {
            // Taffy 0.13 caches ten probes. Give up conservatively if an unusual
            // element uses more; never let a partial log vouch for its caches.
            self.overflowed = true;
            self.probes.clear();
            return;
        }
        self.probes.push((known, available, size));
    }

    fn complete(&self) -> bool {
        !self.overflowed && !self.probes.is_empty()
    }
}

impl Retention {
    fn count(&mut self, f: impl FnOnce(&mut VnRetainedLayoutStats)) {
        if let Some(stats) = self.stats.as_mut() {
            f(stats);
        }
    }

    fn claim(&mut self, scope: Option<GlobalElementId>) -> (Option<Slot>, Option<Entry>) {
        let Some(scope) = scope else {
            return (None, None);
        };
        let Self {
            scopes,
            buckets,
            free_buckets,
            ..
        } = self;
        let bucket = *scopes.entry(scope).or_insert_with(|| {
            if let Some(index) = free_buckets.pop() {
                buckets[index] = Some(ScopeBucket::default());
                index
            } else {
                let index = buckets.len();
                buckets.push(Some(ScopeBucket::default()));
                index
            }
        });
        let scope = buckets[bucket]
            .as_mut()
            .expect("a scope owns its bucket until frame end");
        if scope.claimed_in_frame != self.frame {
            scope.claimed_in_frame = self.frame;
            scope.next_slot = 0;
        }
        let index = scope.next_slot;
        scope.next_slot = index
            .checked_add(1)
            .expect("too many layout requests in one scope");
        if index == scope.entries.len() {
            scope.entries.push(None);
        }
        let previous = scope.entries[index].take();
        if previous.is_none() {
            self.retained_nodes += 1;
        }
        self.claimed_nodes += 1;
        (Some(Slot { bucket, index }), previous)
    }
}

fn remember(
    engine: &mut TaffyLayoutEngine,
    slot: Option<Slot>,
    id: LayoutId,
    log: Option<Rc<RefCell<MeasureLog>>>,
) {
    if let Some(slot) = slot {
        let bucket = engine.retention.buckets[slot.bucket]
            .as_mut()
            .expect("a request's bucket cannot expire before remember");
        debug_assert!(bucket.entries[slot.index].is_none());
        bucket.entries[slot.index] = Some(Entry { id, log });
    } else {
        engine.retention.transient.push(id);
    }
}

fn style(engine: &mut TaffyLayoutEngine, id: LayoutId, requested: &taffy::style::Style) -> bool {
    let previous = engine
        .retention
        .unstretched
        .get(&id)
        .unwrap_or_else(|| engine.taffy.style(id.into()).expect(EXPECT_MESSAGE));
    if previous == requested {
        return false;
    }
    engine.retention.unstretched.remove(&id);
    engine.retention.count(|stats| stats.style_writes += 1);
    engine
        .taffy
        .set_style(id.into(), requested.clone())
        .expect(EXPECT_MESSAGE);
    true
}

fn children(engine: &mut TaffyLayoutEngine, id: LayoutId, requested: &[LayoutId]) {
    for child in requested {
        // A former root can become a child without its requested style changing.
        // Its old viewport override must then stop influencing parent layout.
        if let Some(base) = engine.retention.unstretched.remove(child) {
            if engine.taffy.style((*child).into()).expect(EXPECT_MESSAGE) != &base {
                engine
                    .taffy
                    .set_style((*child).into(), base)
                    .expect(EXPECT_MESSAGE);
            }
        }
    }
    if engine
        .taffy
        .child_ids(id.into())
        .eq(LayoutId::to_taffy_slice(requested).iter().copied())
    {
        return;
    }
    // new_with_children does not detach retained children from their former
    // parents. set_children does, and dirties both changed ancestry chains.
    engine.retention.count(|stats| stats.child_writes += 1);
    engine
        .taffy
        .set_children(id.into(), LayoutId::to_taffy_slice(requested))
        .expect(EXPECT_MESSAGE);
}

pub(super) fn request_layout(
    engine: &mut TaffyLayoutEngine,
    scope: Option<GlobalElementId>,
    requested: taffy::style::Style,
    child_ids: &[LayoutId],
) -> LayoutId {
    let (slot, previous) = engine.retention.claim(scope);
    let prior = previous.as_ref().map(|entry| entry.id);
    engine.retention.count(|stats| {
        if prior.is_some() {
            stats.nodes_reused += 1;
        } else {
            stats.nodes_created += 1;
        }
    });
    let id = if let Some(id) = prior {
        style(engine, id, &requested);
        children(engine, id, child_ids);
        if engine.taffy.get_node_context(id.into()).is_some() {
            engine
                .taffy
                .set_node_context(id.into(), None)
                .expect(EXPECT_MESSAGE);
        }
        id
    } else {
        let id = engine
            .taffy
            .new_leaf(requested)
            .expect(EXPECT_MESSAGE)
            .into();
        children(engine, id, child_ids);
        id
    };
    remember(engine, slot, id, None);
    id
}

pub(super) fn request_measured_layout(
    engine: &mut TaffyLayoutEngine,
    scope: Option<GlobalElementId>,
    requested: taffy::style::Style,
    mut measure: impl FnMut(
        Size<Option<Pixels>>,
        Size<AvailableSpace>,
        &mut Window,
        &mut App,
    ) -> Size<Pixels>
    + 'static,
) -> LayoutId {
    let (slot, previous) = engine.retention.claim(scope);
    let prior = previous.as_ref().map(|entry| entry.id);
    engine.retention.count(|stats| {
        if prior.is_some() {
            stats.nodes_reused += 1;
        } else {
            stats.nodes_created += 1;
        }
    });
    let previous_log = previous.and_then(|entry| entry.log);
    let unchanged_style = prior.is_some_and(|id| !style(engine, id, &requested));
    let replay = unchanged_style
        && previous_log
            .as_ref()
            .is_some_and(|log| log.borrow().complete());
    let log = if replay {
        previous_log.expect("a replay has a complete log")
    } else {
        Rc::default()
    };
    let measure: Box<MeasureFn> = Box::new({
        let log = log.clone();
        move |known, available, window, cx| {
            let size = measure(known, available, window, cx);
            log.borrow_mut().record(known, available, size);
            size
        }
    });
    #[cfg(feature = "stacker")]
    let measure = super::StackSafe::new(measure);
    let id = if let Some(id) = prior {
        children(engine, id, &[]);
        if let Some(context) = engine.taffy.get_node_context_mut(id.into()) {
            // Replacing via set_node_context would dirty every ancestor even
            // when every new measurement is identical to the old one.
            context.measure = measure;
        } else {
            engine
                .taffy
                .set_node_context(id.into(), Some(NodeContext { measure }))
                .expect(EXPECT_MESSAGE);
        }
        if replay {
            engine.retention.pending.push((id, log.clone()));
        } else {
            engine.taffy.mark_dirty(id.into()).expect(EXPECT_MESSAGE);
        }
        id
    } else {
        engine
            .taffy
            .new_leaf_with_context(requested, NodeContext { measure })
            .expect(EXPECT_MESSAGE)
            .into()
    };
    remember(engine, slot, id, Some(log));
    id
}

pub(super) fn note_measure(
    engine: &mut Retention,
    id: NodeId,
    known: Size<Option<Pixels>>,
    available: Size<AvailableSpace>,
) {
    engine.count(|stats| stats.probe_calls += 1);
    record_width(engine, id, known, available);
}

fn record_width(
    engine: &mut Retention,
    id: NodeId,
    known: Size<Option<Pixels>>,
    available: Size<AvailableSpace>,
) {
    let width = known.width.or(match available.width {
        AvailableSpace::Definite(width) => Some(width),
        AvailableSpace::MinContent | AvailableSpace::MaxContent => None,
    });
    engine.measured.insert(id, width);
}

pub(super) fn prepare_compute(engine: &mut TaffyLayoutEngine, scale: f32) {
    let changed = engine
        .retention
        .scale
        .replace(scale)
        .is_some_and(|previous| previous != scale);
    if !changed {
        return;
    }
    // Taffy's cache holds device pixels, whereas measure replay compares
    // logical sizes. Auto/min-content styles and probe keys can stay equal
    // across DPI changes, so ancestor-only invalidation is insufficient.
    for id in engine
        .retention
        .buckets
        .iter()
        .flatten()
        .flat_map(|bucket| bucket.entries.iter().flatten())
        .map(|entry| entry.id)
        .chain(engine.retention.transient.iter().copied())
    {
        engine.taffy.mark_dirty(id.into()).expect(EXPECT_MESSAGE);
    }
}

pub(super) fn replay(engine: &mut TaffyLayoutEngine, window: &mut Window, cx: &mut App) {
    let mut pending = mem::take(&mut engine.retention.pending);
    for (id, log) in pending.drain(..) {
        let probes = log.borrow().probes.clone();
        let Some(context) = engine.taffy.get_node_context_mut(id.into()) else {
            continue;
        };
        let mut last = None;
        let stands = probes.iter().all(|(known, available, expected)| {
            last = Some((*known, *available));
            engine.retention.count(|stats| stats.probe_calls += 1);
            (context.measure)(*known, *available, window, cx) == *expected
        });
        if let Some((known, available)) = last {
            record_width(&mut engine.retention, id.into(), known, available);
        }
        if stands {
            engine.retention.count(|stats| stats.probe_replays += 1);
        } else {
            engine.retention.count(|stats| stats.probe_mismatches += 1);
            // Taffy's old probe caches are now invalid. Start a new log so old
            // constraints cannot make a changed element dirty forever.
            *log.borrow_mut() = MeasureLog::default();
            engine.taffy.mark_dirty(id.into()).expect(EXPECT_MESSAGE);
        }
    }
    engine.retention.pending = pending;
}

pub(super) fn settle(engine: &mut TaffyLayoutEngine, window: &mut Window, cx: &mut App) {
    let mut measured = mem::take(&mut engine.retention.measured);
    let scale = window.scale_factor();
    for (node, last_width) in measured.drain() {
        let layout = engine.taffy.layout(node).expect(EXPECT_MESSAGE);
        let content = Size {
            width: Pixels(
                (layout.size.width
                    - layout.padding.left
                    - layout.padding.right
                    - layout.border.left
                    - layout.border.right)
                    .max(0.)
                    / scale,
            ),
            height: Pixels(
                (layout.size.height
                    - layout.padding.top
                    - layout.padding.bottom
                    - layout.border.top
                    - layout.border.bottom)
                    .max(0.)
                    / scale,
            ),
        };
        if last_width == Some(content.width) {
            continue;
        }
        if let Some(context) = engine.taffy.get_node_context_mut(node) {
            engine.retention.count(|stats| {
                stats.probe_calls += 1;
                stats.settle_calls += 1;
            });
            // A retained Taffy hit can skip the final-width measure after an
            // intrinsic probe. The fresh text element must paint final wraps.
            (context.measure)(
                Size {
                    width: Some(content.width),
                    height: Some(content.height),
                },
                Size {
                    width: AvailableSpace::Definite(content.width),
                    height: AvailableSpace::Definite(content.height),
                },
                window,
                cx,
            );
        }
    }
    engine.retention.measured = measured;
}

pub(super) fn stretch(
    engine: &mut TaffyLayoutEngine,
    id: LayoutId,
    size: Size<Pixels>,
    scale: f32,
) {
    let requested = engine
        .retention
        .unstretched
        .get(&id)
        .unwrap_or_else(|| engine.taffy.style(id.into()).expect(EXPECT_MESSAGE))
        .clone();
    let mut stretched = requested.clone();
    if requested.size.width.is_auto() {
        stretched.size.width = taffy::style::Dimension::length(crate::util::round_to_device_pixel(
            size.width.0,
            scale,
        ));
    }
    if requested.size.height.is_auto() {
        stretched.size.height = taffy::style::Dimension::length(
            crate::util::round_to_device_pixel(size.height.0, scale),
        );
    }
    if stretched == requested && !engine.retention.unstretched.contains_key(&id) {
        return;
    }
    if engine.taffy.style(id.into()).expect(EXPECT_MESSAGE) != &stretched {
        engine.retention.count(|stats| stats.style_writes += 1);
        engine
            .taffy
            .set_style(id.into(), stretched)
            .expect(EXPECT_MESSAGE);
    }
    engine.retention.unstretched.insert(id, requested);
}

fn remove(engine: &mut TaffyLayoutEngine, id: LayoutId) {
    engine.retention.unstretched.remove(&id);
    if let Some(context) = engine.taffy.get_node_context_mut(id.into()) {
        // Taffy 0.13's remove leaves node_context_data alive until slot reuse.
        // Release captured DOM/text/state without dirtying unrelated parents.
        let measure: Box<MeasureFn> = Box::new(|_, _, _, _| Size::default());
        #[cfg(feature = "stacker")]
        let measure = super::StackSafe::new(measure);
        context.measure = measure;
    }
    engine.retention.count(|stats| stats.nodes_freed += 1);
    // Taffy 0.13's remove updates child lists but does not invalidate parents.
    // Do it while the old ancestry still exists, before removal can make a
    // later child-list equality check hide the change.
    if let Some(parent) = engine.taffy.parent(id.into()) {
        engine.taffy.mark_dirty(parent).expect(EXPECT_MESSAGE);
    }
    engine.taffy.remove(id.into()).expect(EXPECT_MESSAGE);
}

pub(super) fn finish_frame(engine: &mut TaffyLayoutEngine) {
    for (id, _) in mem::take(&mut engine.retention.pending) {
        engine.taffy.mark_dirty(id.into()).expect(EXPECT_MESSAGE);
    }
    let mut transient = mem::take(&mut engine.retention.transient);
    for id in transient.drain(..) {
        remove(engine, id);
    }
    engine.retention.transient = transient;
    // In a steady frame every retained slot was claimed. Avoid a second
    // scope lookup or a scan of every node in that common case.
    if engine.retention.retained_nodes != engine.retention.claimed_nodes {
        let retention = &mut engine.retention;
        let mut expired = Vec::new();
        let Retention {
            scopes,
            buckets,
            free_buckets,
            frame,
            ..
        } = retention;
        scopes.retain(|_, index| {
            let bucket_slot = &mut buckets[*index];
            let bucket = bucket_slot.as_mut().expect("a scope owns a live bucket");
            if bucket.claimed_in_frame != *frame {
                expired.extend(bucket.entries.drain(..).flatten().map(|entry| entry.id));
                *bucket_slot = None;
                free_buckets.push(*index);
                return false;
            }
            expired.extend(
                bucket
                    .entries
                    .drain(bucket.next_slot..)
                    .flatten()
                    .map(|entry| entry.id),
            );
            // Keep live scopes proportional to their current slots rather
            // than accumulating each scope's historical high-water mark.
            if bucket.entries.capacity() > bucket.entries.len().saturating_mul(2).max(4) {
                bucket.entries.shrink_to(bucket.entries.len().max(4));
            }
            true
        });
        retention.retained_nodes -= expired.len();
        for id in expired {
            remove(engine, id);
        }
    }
    engine.retention.measured.clear();
    engine.retention.claimed_nodes = 0;
    engine.retention.frame = engine.retention.frame.wrapping_add(1);
}

#[cfg(test)]
mod tests;
