use super::*;
use crate::{
    AnyElement, Bounds, Context, Element, ElementId, InspectorElementId, IntoElement, Render,
    SharedString, StyledText, TestApp, TextLayout, div, prelude::*, px,
};
use std::cell::Cell;

fn scope(name: &'static str) -> GlobalElementId {
    GlobalElementId::new(&[ElementId::Name(name.into())])
}

fn fixed(width: f32, height: f32) -> taffy::style::Style {
    let mut style = taffy::style::Style::default();
    style.size.width = taffy::style::Dimension::length(width);
    style.size.height = taffy::style::Dimension::length(height);
    style
}

fn compute(engine: &mut TaffyLayoutEngine, id: LayoutId) -> f32 {
    engine
        .taffy
        .compute_layout(
            id.into(),
            taffy::geometry::Size {
                width: taffy::style::AvailableSpace::MaxContent,
                height: taffy::style::AvailableSpace::MaxContent,
            },
        )
        .unwrap();
    engine.taffy.layout(id.into()).unwrap().size.width
}

#[test]
fn transient_removal_dirties_auto_parent_before_child_equality_can_hide_it() {
    let mut engine = TaffyLayoutEngine::new();
    engine.enable_retention();
    let child = request_layout(&mut engine, None, fixed(45., 20.), &[]);
    let root = request_layout(
        &mut engine,
        Some(scope("root")),
        Default::default(),
        &[child],
    );
    assert_eq!(compute(&mut engine, root), 45.);
    engine.clear();
    assert!(engine.taffy.dirty(root.into()).unwrap());
    let next_root = request_layout(&mut engine, Some(scope("root")), Default::default(), &[]);
    assert_eq!(next_root, root);
    assert_eq!(compute(&mut engine, next_root), 0.);
    let replacement = request_layout(&mut engine, None, fixed(10., 20.), &[]);
    assert_ne!(
        child, replacement,
        "removed Taffy IDs must retain their generation"
    );
}

#[test]
fn reused_nodes_reconcile_full_style_child_order_and_reparenting() {
    let mut engine = TaffyLayoutEngine::new();
    engine.enable_retention();
    let first = request_layout(&mut engine, Some(scope("first")), fixed(30., 10.), &[]);
    let second = request_layout(&mut engine, Some(scope("second")), fixed(50., 10.), &[]);
    let a = request_layout(
        &mut engine,
        Some(scope("a")),
        Default::default(),
        &[first, second],
    );
    let b = request_layout(&mut engine, Some(scope("b")), Default::default(), &[]);
    assert_eq!(compute(&mut engine, a), 80.);
    assert_eq!(compute(&mut engine, b), 0.);
    engine.clear();

    let first_again = request_layout(&mut engine, Some(scope("first")), fixed(35., 10.), &[]);
    let second_again = request_layout(&mut engine, Some(scope("second")), fixed(50., 10.), &[]);
    assert_eq!(first_again, first);
    assert_eq!(second_again, second);
    // Reparent before requesting a: set_children must detach from its real old parent.
    let b_again = request_layout(
        &mut engine,
        Some(scope("b")),
        Default::default(),
        &[first_again],
    );
    let a_again = request_layout(
        &mut engine,
        Some(scope("a")),
        Default::default(),
        &[second_again],
    );
    assert_eq!(b_again, b);
    assert_eq!(a_again, a);
    assert_eq!(compute(&mut engine, a_again), 50.);
    assert_eq!(compute(&mut engine, b_again), 35.);
    assert_eq!(engine.taffy.parent(first.into()), Some(b.into()));
    engine.clear();

    let first = request_layout(&mut engine, Some(scope("first")), fixed(35., 10.), &[]);
    let second = request_layout(&mut engine, Some(scope("second")), fixed(50., 10.), &[]);
    let a = request_layout(
        &mut engine,
        Some(scope("a")),
        Default::default(),
        &[second, first],
    );
    assert_eq!(compute(&mut engine, a), 85.);
    assert_eq!(engine.taffy.layout(first.into()).unwrap().location.x, 50.);
    engine.clear();
    assert_eq!(
        engine.taffy.total_node_count(),
        3,
        "unclaimed old parent must be collected"
    );
}

#[test]
fn former_auto_root_loses_viewport_stretch_when_attached_as_child() {
    let mut engine = TaffyLayoutEngine::new();
    engine.enable_retention();
    let root = request_layout(
        &mut engine,
        Some(scope("former-root")),
        Default::default(),
        &[],
    );
    stretch(
        &mut engine,
        root,
        Size {
            width: px(300.),
            height: px(100.),
        },
        1.,
    );
    assert_eq!(compute(&mut engine, root), 300.);
    engine.clear();
    let child = request_layout(
        &mut engine,
        Some(scope("former-root")),
        Default::default(),
        &[],
    );
    let parent = request_layout(
        &mut engine,
        Some(scope("new-root")),
        Default::default(),
        &[child],
    );
    assert_eq!(compute(&mut engine, parent), 0.);
}

#[derive(Clone)]
struct Phase {
    text: &'static str,
    width: f32,
    font: f32,
    generation: u32,
    ellipsis: bool,
    rows_a: Vec<(u32, f32)>,
    rows_b: Vec<(u32, f32)>,
    scale: f32,
    rem: f32,
}

impl Default for Phase {
    fn default() -> Self {
        Self {
            text: "alpha beta gamma delta epsilon zeta eta theta",
            width: 100.,
            font: 14.,
            generation: 0,
            ellipsis: false,
            rows_a: vec![(1, 30.), (2, 50.)],
            rows_b: vec![],
            scale: 2.,
            rem: 16.,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Snapshot {
    bounds: Vec<(u32, Bounds<Pixels>)>,
    text: String,
    wrapped: String,
    text_bounds: Bounds<Pixels>,
    measured: (u32, Size<Pixels>, Bounds<Pixels>),
}

struct Capture {
    id: u32,
    inner: AnyElement,
    bounds: Rc<RefCell<Vec<(u32, Bounds<Pixels>)>>>,
}

impl IntoElement for Capture {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}

impl Element for Capture {
    type RequestLayoutState = ();
    type PrepaintState = ();
    fn id(&self) -> Option<ElementId> {
        None
    }
    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }
    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.inner.request_layout(window, cx), ())
    }
    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.bounds.borrow_mut().push((self.id, bounds));
        self.inner.prepaint(window, cx);
    }
    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.inner.paint(window, cx);
    }
}

type MeasuredSnapshot = Rc<Cell<Option<(u32, Size<Pixels>, Bounds<Pixels>)>>>;

struct FreshMeasured {
    generation: u32,
    fresh: Rc<Cell<Option<(u32, Size<Pixels>)>>>,
    painted: MeasuredSnapshot,
}

impl IntoElement for FreshMeasured {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}

impl Element for FreshMeasured {
    type RequestLayoutState = ();
    type PrepaintState = ();
    fn id(&self) -> Option<ElementId> {
        Some("fresh-measured".into())
    }
    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }
    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        _: &mut App,
    ) -> (LayoutId, ()) {
        let fresh = self.fresh.clone();
        let generation = self.generation;
        let layout = window.request_measured_layout(Default::default(), move |_, _, _, _| {
            let size = Size {
                width: px(24.),
                height: px(12.),
            };
            fresh.set(Some((generation, size)));
            size
        });
        (layout, ())
    }
    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut Window,
        _: &mut App,
    ) {
    }
    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        _: &mut Window,
        _: &mut App,
    ) {
        let (generation, size) = self
            .fresh
            .get()
            .expect("the CURRENT element's measurement closure must run before paint");
        assert_eq!(generation, self.generation);
        self.painted.set(Some((generation, size, bounds)));
    }
}

struct Fixture {
    phase: Phase,
    layout: Option<TextLayout>,
    bounds: Rc<RefCell<Vec<(u32, Bounds<Pixels>)>>>,
    measured: MeasuredSnapshot,
}

impl Fixture {
    fn snapshot(&self) -> Snapshot {
        let layout = self.layout.as_ref().unwrap();
        let mut bounds = self.bounds.borrow().clone();
        bounds.sort_by_key(|(id, _)| *id);
        Snapshot {
            bounds,
            text: layout.text(),
            wrapped: layout.wrapped_text(),
            text_bounds: layout.bounds(),
            measured: self.measured.get().unwrap(),
        }
    }
}

impl Render for Fixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.bounds.borrow_mut().clear();
        self.measured.set(None);
        let text = StyledText::new(SharedString::from(self.phase.text));
        self.layout = Some(text.layout().clone());
        let capture = |id, inner| Capture {
            id,
            inner,
            bounds: self.bounds.clone(),
        };
        let row = |id, rows: &[(u32, f32)]| {
            capture(
                id,
                div()
                    .id(id as usize)
                    .flex()
                    .items_start()
                    .children(rows.iter().map(|(key, width)| {
                        capture(
                            *key,
                            div()
                                .id(*key as usize)
                                .w(px(*width))
                                .h(px(10.))
                                .into_any_element(),
                        )
                    }))
                    .into_any_element(),
            )
        };
        div()
            .id("fixture")
            .size_full()
            .flex()
            .flex_col()
            .items_start()
            .child(capture(
                100,
                div()
                    .id("text-box")
                    .w(px(self.phase.width))
                    .text_size(px(self.phase.font))
                    .when(self.phase.ellipsis, |element| element.truncate())
                    .child(text)
                    .into_any_element(),
            ))
            .child(row(200, &self.phase.rows_a))
            .child(row(300, &self.phase.rows_b))
            .child(FreshMeasured {
                generation: self.phase.generation,
                fresh: Rc::default(),
                painted: self.measured.clone(),
            })
    }
}

fn draw_phases(retained: bool, phases: &[Phase]) -> (Vec<Snapshot>, VnRetainedLayoutStats) {
    let mut app = TestApp::new();
    let mut window = app.open_window(|window, _| {
        if retained {
            window.vn_enable_retained_layout();
            window.vn_profile_retained_layout(true);
        }
        Fixture {
            phase: Phase::default(),
            layout: None,
            bounds: Rc::default(),
            measured: Rc::default(),
        }
    });
    window.draw();
    let mut snapshots = Vec::new();
    let mut last_scale = 2.;
    for phase in phases {
        if phase.scale != last_scale {
            window.simulate_scale_factor_change(phase.scale);
            last_scale = phase.scale;
        }
        window.update(|view, window, cx| {
            view.phase = phase.clone();
            window.set_rem_size(px(phase.rem));
            cx.notify();
        });
        window.draw();
        snapshots.push(window.read(|view, _| view.snapshot()));
    }
    let stats = window
        .update(|_, window, _| window.vn_take_retained_layout_stats())
        .unwrap_or_default();
    drop(window);
    app.update(|cx| cx.shutdown());
    (snapshots, stats)
}

#[test]
fn repeated_real_draws_refresh_measurements_and_match_text_wrap_geometry() {
    let base = Phase::default();
    let phases = vec![
        base.clone(),
        Phase {
            generation: 1,
            ..base.clone()
        },
        Phase {
            text: "equal width text can still be entirely new",
            generation: 2,
            ..base.clone()
        },
        Phase {
            width: 55.,
            generation: 3,
            ..base.clone()
        },
        Phase {
            width: 200.,
            font: 20.,
            generation: 4,
            ..base.clone()
        },
        Phase {
            width: 60.,
            ellipsis: true,
            generation: 5,
            ..base.clone()
        },
        Phase {
            width: 125.,
            scale: 1.25,
            rem: 19.,
            generation: 6,
            ..base.clone()
        },
    ];
    let (reference, _) = draw_phases(false, &phases);
    let (actual, stats) = draw_phases(true, &phases);
    assert_eq!(actual, reference);
    assert!(stats.nodes_reused > 0);
    assert!(
        stats.probe_replays > 0,
        "warm cached geometry must use current measurement state"
    );
    assert!(
        stats.probe_mismatches > 0,
        "changed text/constraints must invalidate stale cached measures"
    );
    assert_ne!(
        actual[0].wrapped, actual[3].wrapped,
        "the width phase must actually change wrapping"
    );
}

#[test]
fn real_draws_match_after_style_reorder_remove_and_parent_changes() {
    let base = Phase::default();
    let phases = vec![
        base.clone(),
        Phase {
            rows_a: vec![(2, 50.), (1, 35.)],
            generation: 1,
            ..base.clone()
        },
        Phase {
            rows_a: vec![(2, 50.)],
            rows_b: vec![(1, 35.)],
            generation: 2,
            ..base.clone()
        },
        Phase {
            rows_a: vec![],
            rows_b: vec![(1, 75.)],
            generation: 3,
            ..base.clone()
        },
        Phase {
            rows_a: vec![(3, 20.)],
            rows_b: vec![],
            generation: 4,
            ..base
        },
    ];
    let (reference, _) = draw_phases(false, &phases);
    let (actual, stats) = draw_phases(true, &phases);
    assert_eq!(actual, reference);
    assert!(stats.style_writes > 0);
    assert!(stats.child_writes > 0);
    assert!(stats.nodes_freed > 0);
}

#[test]
fn scope_buckets_keep_full_paths_and_independent_interleaved_slots() {
    let mut engine = TaffyLayoutEngine::new();
    engine.enable_retention();
    let first_scope = GlobalElementId::new(&["first-parent".into(), "same-leaf".into()]);
    let second_scope = GlobalElementId::new(&["second-parent".into(), "same-leaf".into()]);
    let first = request_layout(&mut engine, Some(first_scope.clone()), fixed(10., 10.), &[]);
    let second = request_layout(
        &mut engine,
        Some(second_scope.clone()),
        fixed(20., 10.),
        &[],
    );
    let first_extra = request_layout(&mut engine, Some(first_scope.clone()), fixed(30., 10.), &[]);
    assert_ne!(first, second);
    assert_ne!(first, first_extra);
    assert_eq!(engine.retention.scopes.len(), 2);
    engine.clear();

    // Change the interleaving while retaining each scope's own request order.
    let second_again = request_layout(
        &mut engine,
        Some(second_scope.clone()),
        fixed(20., 10.),
        &[],
    );
    let first_again = request_layout(&mut engine, Some(first_scope.clone()), fixed(10., 10.), &[]);
    let first_extra_again =
        request_layout(&mut engine, Some(first_scope.clone()), fixed(30., 10.), &[]);
    assert_eq!(second_again, second);
    assert_eq!(first_again, first);
    assert_eq!(first_extra_again, first_extra);
    engine.clear();
    assert_eq!(engine.retention.retained_nodes, 3);
    assert_eq!(engine.taffy.total_node_count(), 3);
}

#[test]
fn scope_bucket_tail_and_identity_expiry_recycle_without_retaining_old_nodes() {
    let mut engine = TaffyLayoutEngine::new();
    engine.enable_retention();
    let first_scope = scope("first");
    let second_scope = scope("second");
    let first = request_layout(&mut engine, Some(first_scope.clone()), fixed(10., 10.), &[]);
    let dropped_tail = request_layout(&mut engine, Some(first_scope.clone()), fixed(30., 10.), &[]);
    for _ in 0..30 {
        request_layout(&mut engine, Some(first_scope.clone()), fixed(30., 10.), &[]);
    }
    let second = request_layout(
        &mut engine,
        Some(second_scope.clone()),
        fixed(20., 10.),
        &[],
    );
    let first_bucket = engine.retention.scopes[&first_scope];
    engine.clear();

    assert_eq!(
        request_layout(&mut engine, Some(first_scope.clone()), fixed(10., 10.), &[]),
        first
    );
    assert_eq!(
        request_layout(
            &mut engine,
            Some(second_scope.clone()),
            fixed(20., 10.),
            &[]
        ),
        second
    );
    engine.clear();
    assert_eq!(engine.retention.retained_nodes, 2);
    assert_eq!(
        engine.retention.buckets[first_bucket]
            .as_ref()
            .unwrap()
            .entries
            .len(),
        1
    );
    assert_eq!(engine.taffy.total_node_count(), 2);
    assert!(
        engine.retention.buckets[first_bucket]
            .as_ref()
            .unwrap()
            .entries
            .capacity()
            <= 4,
        "a surviving scope must release a large expired slot tail"
    );

    request_layout(
        &mut engine,
        Some(second_scope.clone()),
        fixed(20., 10.),
        &[],
    );
    engine.clear();
    assert!(!engine.retention.scopes.contains_key(&first_scope));
    assert_eq!(engine.retention.retained_nodes, 1);
    assert_eq!(engine.retention.free_buckets, [first_bucket]);

    let replacement_scope = scope("replacement");
    let replacement = request_layout(
        &mut engine,
        Some(replacement_scope.clone()),
        fixed(40., 10.),
        &[],
    );
    assert_ne!(replacement, first);
    assert_ne!(replacement, dropped_tail);
    assert_eq!(engine.retention.scopes[&replacement_scope], first_bucket);
    request_layout(&mut engine, Some(second_scope), fixed(20., 10.), &[]);
    engine.clear();
    assert_eq!(
        engine.retention.buckets.len(),
        2,
        "recycled scopes cannot grow the arena"
    );
    assert_eq!(engine.retention.retained_nodes, 2);
    assert_eq!(engine.taffy.total_node_count(), 2);
}
