//! Optional drawing-thread spans supplied by an embedding application.

use std::{
    cell::Cell,
    marker::PhantomData,
    panic::Location,
    rc::Rc,
    sync::{
        RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::PlatformWindow;
use raw_window_handle::RawWindowHandle;

thread_local! {
    static CURRENT_TARGET: Cell<usize> = const { Cell::new(0) };
}

/// The target is an opaque native window key (HWND on Windows).
/// The callbacks run on the drawing thread and must not invoke UI or JS code.
#[derive(Clone, Copy)]
pub struct FrameProfilerCallbacks {
    /// Starts a span and returns zero when the target is not being recorded.
    pub begin: fn(usize, &'static str, &'static str, &'static str, u32, u32) -> u64,
    /// Ends a nonzero token on the same thread that started it.
    pub end: fn(u64),
}
static ENABLED: AtomicBool = AtomicBool::new(false);
static CALLBACKS: RwLock<Option<FrameProfilerCallbacks>> = RwLock::new(None);

/// Installs or removes optional CPU span hooks without linking an embedder into GPUI.
pub fn set_frame_profiler(callbacks: Option<FrameProfilerCallbacks>) {
    ENABLED.store(false, Ordering::Relaxed);
    let enabled = callbacks.is_some();
    *CALLBACKS.write().unwrap_or_else(|error| error.into_inner()) = callbacks;
    ENABLED.store(enabled, Ordering::Relaxed);
}
/// A drawing-thread span that ends when dropped.
#[doc(hidden)]
pub struct Span {
    token: u64,
    end: Option<fn(u64)>,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(end) = self.end {
            if self.token != 0 {
                end(self.token);
            }
        }
    }
}
#[track_caller]
#[inline]
#[doc(hidden)]
pub fn span(target: usize, category: &'static str, name: &'static str) -> Span {
    if !ENABLED.load(Ordering::Relaxed) {
        return Span {
            token: 0,
            end: None,
            _thread: PhantomData,
        };
    }
    let callbacks = *CALLBACKS.read().unwrap_or_else(|error| error.into_inner());
    let Some(callbacks) = callbacks else {
        return Span {
            token: 0,
            end: None,
            _thread: PhantomData,
        };
    };
    let caller = Location::caller();
    Span {
        token: (callbacks.begin)(
            target,
            category,
            name,
            caller.file(),
            caller.line(),
            caller.column(),
        ),
        end: Some(callbacks.end),
        _thread: PhantomData,
    }
}

/// Starts a span for the window whose GPUI draw is active on this thread.
/// Disabled hooks do not access thread-local state or read the clock.
#[track_caller]
#[inline]
pub fn current_span(category: &'static str, name: &'static str) -> Span {
    if !ENABLED.load(Ordering::Relaxed) {
        return Span {
            token: 0,
            end: None,
            _thread: PhantomData,
        };
    }
    span(CURRENT_TARGET.with(Cell::get), category, name)
}

pub(crate) struct WindowScope {
    previous: Option<usize>,
    _thread: PhantomData<Rc<()>>,
}

impl Drop for WindowScope {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            CURRENT_TARGET.with(|target| target.set(previous));
        }
    }
}

fn target_scope(target: usize) -> WindowScope {
    WindowScope {
        previous: Some(CURRENT_TARGET.with(|current| current.replace(target))),
        _thread: PhantomData,
    }
}

#[inline]
pub(crate) fn window_scope(window: &dyn PlatformWindow) -> WindowScope {
    if !ENABLED.load(Ordering::Relaxed) {
        return WindowScope {
            previous: None,
            _thread: PhantomData,
        };
    }
    let target = match window.window_handle().ok().map(|handle| handle.as_raw()) {
        Some(RawWindowHandle::Win32(handle)) => handle.hwnd.get() as usize,
        _ => 0,
    };
    target_scope(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static EVENTS: Mutex<Vec<(usize, &'static str)>> = Mutex::new(Vec::new());

    fn begin(
        target: usize,
        _: &'static str,
        name: &'static str,
        _: &'static str,
        _: u32,
        _: u32,
    ) -> u64 {
        EVENTS.lock().unwrap().push((target, name));
        target as u64
    }

    fn end(token: u64) {
        EVENTS.lock().unwrap().push((token as usize, "end"));
    }

    #[test]
    fn spans_follow_nested_window_scopes_and_ignore_disabled_or_zero_tokens() {
        set_frame_profiler(None);
        EVENTS.lock().unwrap().clear();
        drop(current_span("native", "disabled"));
        assert!(EVENTS.lock().unwrap().is_empty());

        set_frame_profiler(Some(FrameProfilerCallbacks { begin, end }));
        {
            let _outer = target_scope(11);
            drop(current_span("native", "outer"));
            {
                let _inner = target_scope(22);
                drop(current_span("native", "inner"));
            }
            drop(current_span("native", "restored"));
        }
        drop(current_span("native", "outside"));
        set_frame_profiler(None);
        assert_eq!(
            *EVENTS.lock().unwrap(),
            vec![
                (11, "outer"),
                (11, "end"),
                (22, "inner"),
                (22, "end"),
                (11, "restored"),
                (11, "end"),
                (0, "outside"),
            ]
        );
    }
}
