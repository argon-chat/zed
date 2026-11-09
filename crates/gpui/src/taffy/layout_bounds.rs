// Copyright 2022 - 2025 Zed Industries, Inc.
// Copyright 2026 Longbridge.
// SPDX-License-Identifier: Apache-2.0
// Adapted from longbridge/gpui-fast, commit fb8cf9413777e43e7c38bf5917647ee984453f37,
// crates/gpui/src/fast/layout_bounds.rs. Geometry remains the host fork's.

//! The tables [`TaffyLayoutEngine::layout_bounds`] caches absolute bounds in,
//! indexed by the node's Taffy slot instead of hashed.
//!
//! Every element asks for its bounds in prepaint, thousands of them a frame,
//! and each first query of a node looks its parent up, then stores the node's
//! origin and bounds. Upstream keeps both in `FxHashMap<LayoutId, _>`s and
//! clears them at the end of every frame. [`LayoutIdMap`] answers the same
//! calls from a `Vec` indexed by the slot half of the node id, and forgets
//! everything at the end of a frame by starting a new pass instead of
//! clearing. What `layout_bounds` computes, pixel snapping included, is left
//! exactly as upstream wrote it.
//!
//! [`TaffyLayoutEngine::layout_bounds`]: crate::taffy::TaffyLayoutEngine::layout_bounds

use crate::LayoutId;

/// One slot of a [`LayoutIdMap`].
#[derive(Clone, Copy, Default)]
struct Entry<V> {
    /// The whole node id, slot and version, since Taffy gives a removed node's
    /// slot to the next node it creates.
    node: u64,
    /// The pass the value was stored in. It stands only in the current one.
    pass: u32,
    value: V,
}

/// A map from layout node to `V`, for the few calls `layout_bounds` and
/// `compute_layout` make of their caches: `get`, `insert`, `remove` and
/// `clear`, as on the `FxHashMap` it replaces.
pub(crate) struct LayoutIdMap<V> {
    /// Indexed by the slot of the node's id. Slots are dense, as Taffy reuses
    /// those of removed nodes.
    entries: Vec<Entry<V>>,
    /// Entries of any other pass are absent. Never 0, the pass of an entry that
    /// was never stored or has been removed.
    pass: u32,
}

impl<V> Default for LayoutIdMap<V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            pass: 1,
        }
    }
}

/// The id Taffy gave the node: a slotmap key's `as_ffi` value, the version in
/// the high half and the slot in the low.
#[inline]
fn raw(id: &LayoutId) -> u64 {
    u64::from(taffy::NodeId::from(*id))
}

#[inline]
fn slot(raw: u64) -> usize {
    raw as u32 as usize
}

impl<V: Copy + Default> LayoutIdMap<V> {
    #[inline]
    pub(crate) fn get(&self, id: &LayoutId) -> Option<&V> {
        let raw = raw(id);
        self.entries
            .get(slot(raw))
            .filter(|entry| entry.pass == self.pass && entry.node == raw)
            .map(|entry| &entry.value)
    }

    #[inline]
    pub(crate) fn insert(&mut self, id: LayoutId, value: V) {
        let raw = raw(&id);
        let slot = slot(raw);
        if slot >= self.entries.len() {
            self.entries.resize(slot + 1, Entry::default());
        }
        self.entries[slot] = Entry {
            node: raw,
            pass: self.pass,
            value,
        };
    }

    #[inline]
    pub(crate) fn remove(&mut self, id: &LayoutId) {
        let raw = raw(id);
        if let Some(entry) = self.entries.get_mut(slot(raw))
            && entry.node == raw
        {
            entry.pass = 0;
        }
    }

    /// Forgets every entry, at the end of a frame, without touching them.
    pub(crate) fn clear(&mut self) {
        self.pass = self.pass.wrapping_add(1);
        if self.pass == 0 {
            // Four billion frames on, entries stored in pass 1 would be back.
            self.entries.fill(Entry::default());
            self.pass = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LayoutIdMap;
    use crate::LayoutId;
    use taffy::{TaffyTree, style::Style};

    #[test]
    fn entries_last_one_pass_and_belong_to_one_node_of_a_slot() {
        let mut taffy: TaffyTree<()> = TaffyTree::new();
        let first: LayoutId = taffy.new_leaf(Style::default()).unwrap().into();
        let mut map = LayoutIdMap::<u32>::default();
        map.insert(first, 7);
        assert_eq!(map.get(&first), Some(&7));

        // A node given the removed node's slot does not see its entry.
        taffy.remove(first.into()).unwrap();
        let second: LayoutId = taffy.new_leaf(Style::default()).unwrap().into();
        assert_ne!(first, second);
        assert_eq!(map.get(&second), None);
        map.remove(&second);
        assert_eq!(map.get(&first), Some(&7));

        map.insert(second, 8);
        assert_eq!(map.get(&second), Some(&8));
        assert_eq!(map.get(&first), None);
        map.remove(&first);
        assert_eq!(
            map.get(&second),
            Some(&8),
            "a stale version must not remove the new slot occupant"
        );
        map.remove(&second);
        assert_eq!(map.get(&second), None);

        map.insert(second, 9);
        map.clear();
        assert_eq!(map.get(&second), None);
        map.insert(second, 10);
        assert_eq!(map.get(&second), Some(&10));
    }

    #[test]
    fn pass_wrap_does_not_resurrect_entries_and_keeps_allocated_capacity() {
        let mut taffy: TaffyTree<()> = TaffyTree::new();
        let node: LayoutId = taffy.new_leaf(Style::default()).unwrap().into();
        let mut map = LayoutIdMap::<u32>::default();
        map.insert(node, 17);
        let capacity = map.entries.capacity();
        map.pass = u32::MAX;
        map.clear();
        assert_eq!(map.pass, 1);
        assert_eq!(
            map.get(&node),
            None,
            "pass one from before the wrap is stale"
        );
        assert_eq!(map.entries.capacity(), capacity);
        map.insert(node, 23);
        map.clear();
        assert_eq!(map.get(&node), None);
        assert_eq!(map.entries.capacity(), capacity);
    }

    fn fractional_tree() -> (super::super::TaffyLayoutEngine, LayoutId) {
        use taffy::style::{Dimension, LengthPercentageAuto, Position};
        let mut engine = super::super::TaffyLayoutEngine::new();
        let mut child_style = Style::default();
        child_style.position = Position::Absolute;
        child_style.inset.left = LengthPercentageAuto::length(4.375);
        child_style.inset.top = LengthPercentageAuto::length(1.125);
        child_style.size.width = Dimension::length(27.75);
        child_style.size.height = Dimension::length(13.25);
        let child = engine.taffy.new_leaf(child_style).unwrap();
        let mut parent_style = Style::default();
        parent_style.position = Position::Absolute;
        parent_style.inset.left = LengthPercentageAuto::length(11.25);
        parent_style.inset.top = LengthPercentageAuto::length(23.5);
        parent_style.size.width = Dimension::length(101.25);
        parent_style.size.height = Dimension::length(54.5);
        let parent = engine
            .taffy
            .new_with_children(parent_style, &[child])
            .unwrap();
        let mut root_style = Style::default();
        root_style.size.width = Dimension::length(400.);
        root_style.size.height = Dimension::length(200.);
        let root = engine
            .taffy
            .new_with_children(root_style, &[parent])
            .unwrap();
        engine
            .taffy
            .compute_layout(
                root,
                taffy::geometry::Size {
                    width: taffy::style::AvailableSpace::MaxContent,
                    height: taffy::style::AvailableSpace::MaxContent,
                },
            )
            .unwrap();
        (engine, child.into())
    }

    #[test]
    fn cached_fractional_bounds_preserve_absolute_edge_rounding_at_hidpi() {
        use crate::{Bounds, point, px, size};
        let (mut engine, child) = fractional_tree();
        for scale in [1.0, 1.25, 1.5, 2.0] {
            engine.absolute_layout_bounds.clear();
            engine.absolute_outer_origins.clear();
            // Raw device origin is (15.625,24.625), far corner (43.375,37.875).
            // Rounding each ancestor first would give a different child origin.
            let expected = Bounds::new(
                point(px(16. / scale), px(25. / scale)),
                size(px(27. / scale), px(13. / scale)),
            );
            assert_eq!(engine.layout_bounds(child, scale), expected);
            assert_eq!(
                engine.layout_bounds(child, scale),
                expected,
                "the cached read changed geometry"
            );
        }
    }

    #[test]
    fn cached_bounds_keep_fractional_affine_paint_coordinates() {
        use crate::{Bounds, ScaledPixels, TransformationMatrix, point, size};
        let (mut engine, child) = fractional_tree();
        let bounds = engine.layout_bounds(child, 1.0);
        assert_eq!(engine.layout_bounds(child, 1.0), bounds);
        let transform = TransformationMatrix {
            rotation_scale: [[-1.5, 0.25], [-0.5, 2.0]],
            translation: [-7.125, 3.875],
        };
        let transformed =
            crate::scene::transformed_bounds(bounds.map(|value| ScaledPixels(value.0)), transform);
        assert_eq!(
            transformed,
            Bounds::new(
                point(ScaledPixels(-65.375), ScaledPixels(32.375)),
                size(ScaledPixels(43.75), ScaledPixels(39.5)),
            )
        );
        assert_eq!(
            engine.layout_bounds(child, 1.0),
            bounds,
            "paint transforms must not enter layout coordinates"
        );
    }
}
