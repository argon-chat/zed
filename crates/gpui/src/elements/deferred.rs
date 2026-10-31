use crate::{
    AnyElement, App, Bounds, Element, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    Pixels, Window,
};
use std::rc::Rc;

/// Builds a `Deferred` element, which delays the layout and paint of its child.
pub fn deferred(child: impl IntoElement) -> Deferred {
    Deferred {
        child: Some(child.into_any_element()),
        priority: 0,
        clip_to_ancestors: false,
        stack_key: None,
    }
}

/// An element which delays the painting of its child until after all of
/// its ancestors, while keeping its layout as part of the current element tree.
pub struct Deferred {
    child: Option<AnyElement>,
    priority: usize,
    clip_to_ancestors: bool,
    stack_key: Option<Rc<[u128]>>,
}

impl Deferred {
    /// Sets the `priority` value of the `deferred` element, which
    /// determines the drawing order relative to other deferred elements,
    /// with higher values being drawn on top.
    pub fn with_priority(mut self, priority: usize) -> Self {
        self.priority = priority;
        self
    }

    /// Keep the CONTENT MASK in force where the element was deferred from.
    ///
    /// A deferred draw is replayed at the top of the frame, so by default it
    /// escapes every ancestor's `overflow` clip along with their paint order —
    /// which is what a popover or a context menu wants. CSS says something
    /// narrower: a positioned element with a `z-index` escapes its siblings'
    /// PAINT ORDER but is still clipped by the overflow of the ancestors it is
    /// laid out inside. This opts into that reading.
    ///
    /// (Without it, a `z-index`ed page section inside a scrolling viewport
    /// paints its overflow outside the viewport entirely — over whatever the
    /// host draws below it.)
    pub fn clip_to_ancestors(mut self, clip: bool) -> Self {
        self.clip_to_ancestors = clip;
        self
    }

    /// vue-native fork: where this draw sorts among the deferred draws of the
    /// same `priority`, compared lexicographically, a key that is a prefix of
    /// another sorting first. Draws without a key sort before every keyed one.
    ///
    /// Unlike `priority`, which is one number, a key can say where a draw sits
    /// inside the draws it is nested in: an embedder whose paint order is a
    /// tree of layers (CSS stacking contexts) gives each draw the path of its
    /// layers, and the whole of an earlier layer then paints before a later
    /// one however deep its own draws are nested. The order is the PAINT order
    /// and the hit-test order alike (`Frame::vn_hit_order`).
    pub fn with_stack_key(mut self, key: Rc<[u128]>) -> Self {
        self.stack_key = Some(key);
        self
    }
}

impl Element for Deferred {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let layout_id = self.child.as_mut().unwrap().request_layout(window, cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        _cx: &mut App,
    ) {
        let child = self.child.take().unwrap();
        let element_offset = window.element_offset();
        let mask = self.clip_to_ancestors.then(|| window.content_mask());
        window.defer_draw_with_stack_key(
            child,
            element_offset,
            self.priority,
            mask,
            self.stack_key.take(),
        )
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }
}

impl IntoElement for Deferred {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Deferred {
    /// Sets a priority for the element. A higher priority conceptually means painting the element
    /// on top of deferred draws with a lower priority (i.e. closer to the viewer).
    pub fn priority(mut self, priority: usize) -> Self {
        self.priority = priority;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Context, Entity, StyleRefinement, TestAppContext, Window, anchored, deferred, div, point,
        prelude::*, px, size,
    };

    /// A stand-in for a dock panel hosting a popover (deferred draw) whose
    /// content opens another popover (a deferred draw created while
    /// prepainting the first one's content).
    struct PanelView;

    impl Render for PanelView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().key_context("Panel").size_full().child(
                deferred(
                    anchored().position(point(px(10.), px(10.))).child(
                        div().key_context("Popover").w(px(200.)).h(px(200.)).child(
                            deferred(
                                anchored().position(point(px(30.), px(30.))).child(
                                    div()
                                        .key_context("NestedMenu")
                                        .debug_selector(|| "NESTED_MENU".into())
                                        .w(px(50.))
                                        .h(px(50.)),
                                ),
                            )
                            .with_priority(2),
                        ),
                    ),
                )
                .with_priority(1),
            )
        }
    }

    struct RootView {
        panel: Entity<PanelView>,
    }

    impl Render for RootView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().key_context("Root").size_full().child(
                self.panel
                    .clone()
                    .cached(StyleRefinement::default().size_full()),
            )
        }
    }

    /// Regression test for a crash with nested deferred draws (e.g. a popover
    /// menu inside a popover hosted by a cached dock panel). Prepaint indices
    /// recorded during the deferred draw rounds must index the same
    /// `deferred_draws` vector that `reuse_prepaint` slices on the next frame;
    /// previously they were measured against a transient per-round vector, so
    /// reusing the panel's subtree grafted the wrong deferred draws and
    /// panicked in the dispatch tree.
    #[gpui::test]
    fn test_nested_deferred_draws_with_reused_views(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(800.), px(600.)), |_, cx| {
            let panel = cx.new(|_| PanelView);
            RootView { panel }
        });
        cx.run_until_parked();

        let menu_bounds = window
            .update(cx, |_, window, _| {
                window
                    .rendered_frame
                    .debug_bounds
                    .get("NESTED_MENU")
                    .copied()
            })
            .unwrap()
            .expect("NESTED_MENU debug bounds not found");
        assert_eq!(menu_bounds.size, size(px(50.), px(50.)));

        // Re-render only the root view; the panel is cached, so its subtree -
        // including both deferred draw records - is reused from the previous
        // frame.
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        // Reuse the subtree a second time, exercising ranges that were
        // themselves recorded during a reused frame.
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        // Re-render the panel itself again to prove the popovers still draw.
        window
            .update(cx, |root, _, cx| {
                root.panel.update(cx, |_, cx| cx.notify());
            })
            .unwrap();
        cx.run_until_parked();

        window
            .update(cx, |_, window, _| {
                assert_eq!(window.rendered_frame.deferred_draws.len(), 2);
                assert!(
                    window
                        .rendered_frame
                        .debug_bounds
                        .contains_key("NESTED_MENU")
                );
            })
            .unwrap();
    }
}

/// vue-native fork: draws keyed with `with_stack_key` paint, and hit-test, in
/// key order, however deep each one is nested; and a draw keeps the opacity
/// in force where it was deferred from.
#[cfg(test)]
mod vn_stack_key_tests {
    use crate::{
        Context, HitboxBehavior, HitboxId, TestAppContext, Window, canvas, deferred, div, point,
        prelude::*, px, size,
    };
    use std::{cell::RefCell, rc::Rc};

    type Log = Rc<RefCell<Vec<(&'static str, HitboxId, f32)>>>;

    fn key(slots: &[u128]) -> Rc<[u128]> {
        slots.into()
    }

    /// A 100px square at the window's origin with a hitbox, logging its paint.
    fn probe(name: &'static str, log: &Log) -> impl IntoElement {
        let log = log.clone();
        canvas(
            |bounds, window, _| window.insert_hitbox(bounds, HitboxBehavior::Normal),
            move |_, hitbox, window, _| {
                log.borrow_mut().push((name, hitbox.id, window.element_opacity))
            },
        )
        .absolute()
        .top_0()
        .left_0()
        .size(px(100.))
    }

    /// Two overlays of the same z, the EARLIER one with a layer inside it
    /// that is deferred one round deeper: an embedder's popover with a
    /// `z-index: 2` layer, and the menu it opens. CSS paints the popover
    /// whole, layer included, before the menu.
    struct TwoOverlays {
        log: Log,
    }

    impl Render for TwoOverlays {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let log = &self.log;
            div()
                .size_full()
                .child(
                    deferred(
                        div()
                            .absolute()
                            .top_0()
                            .left_0()
                            .size(px(100.))
                            .child(probe("popover", log))
                            .child(deferred(probe("layer", log)).with_stack_key(key(&[50, 2]))),
                    )
                    .with_stack_key(key(&[50])),
                )
                .child(deferred(probe("menu", log)).with_stack_key(key(&[51])))
        }
    }

    /// The last frame's paints, in order.
    fn last_frame(log: &Log, n: usize) -> Vec<(&'static str, HitboxId, f32)> {
        let log = log.borrow();
        assert!(log.len() >= n, "painted: {:?}", log.iter().map(|e| e.0).collect::<Vec<_>>());
        log[log.len() - n..].to_vec()
    }

    #[gpui::test]
    fn a_nested_draw_paints_and_hit_tests_inside_its_overlay(cx: &mut TestAppContext) {
        let log: Log = Default::default();
        let view_log = log.clone();
        let window = cx.open_window(size(px(200.), px(200.)), move |_, _| TwoOverlays {
            log: view_log,
        });
        cx.run_until_parked();
        let frame = last_frame(&log, 3);
        let names: Vec<_> = frame.iter().map(|e| e.0).collect();
        // Rounds alone painted ["popover", "menu", "layer"].
        assert_eq!(names, ["popover", "layer", "menu"]);
        let id = |name| frame.iter().find(|e| e.0 == name).unwrap().1;
        window
            .update(cx, |_, window, _| {
                let hit = window.rendered_frame.hit_test(point(px(50.), px(50.)), window.scale_factor());
                assert_eq!(
                    hit.ids.to_vec(),
                    vec![id("menu"), id("layer"), id("popover")],
                    "topmost painted first"
                );
            })
            .unwrap();
    }

    struct Faded {
        log: Log,
    }

    impl Render for Faded {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .opacity(0.5)
                .child(deferred(probe("inside", &self.log)).with_stack_key(key(&[1])))
        }
    }

    #[gpui::test]
    fn a_deferred_draw_keeps_its_ancestors_opacity(cx: &mut TestAppContext) {
        let log: Log = Default::default();
        let view_log = log.clone();
        let _window = cx.open_window(size(px(200.), px(200.)), move |_, _| Faded { log: view_log });
        cx.run_until_parked();
        assert_eq!(last_frame(&log, 1)[0].2, 0.5);
    }
}

/// vue-native fork: a deferred element paints under the text style its
/// ancestors set, as it is prepainted under it.
#[cfg(test)]
mod vn_deferred_text_style_tests {
    use crate::{
        Context, Hsla, TestAppContext, Window, canvas, deferred, div, prelude::*, px, red, size,
    };
    use std::{cell::RefCell, rc::Rc};

    struct Tinted {
        seen: Rc<RefCell<Vec<Hsla>>>,
    }

    impl Render for Tinted {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let seen = self.seen.clone();
            div().size_full().text_color(red()).child(
                deferred(
                    canvas(
                        |_, _, _| (),
                        move |_, _, window, _| seen.borrow_mut().push(window.text_style().color),
                    )
                    .size(px(10.)),
                )
                .with_priority(1),
            )
        }
    }

    #[gpui::test]
    fn a_deferred_paint_reads_the_inherited_text_style(cx: &mut TestAppContext) {
        let seen = Rc::new(RefCell::new(Vec::new()));
        let view_seen = seen.clone();
        let _window = cx.open_window(size(px(200.), px(200.)), move |_, _| Tinted { seen: view_seen });
        cx.run_until_parked();
        let seen = seen.borrow();
        assert!(!seen.is_empty(), "the deferred canvas painted");
        assert!(seen.iter().all(|c| *c == red()), "painted under the ancestor's colour: {seen:?}");
    }
}
