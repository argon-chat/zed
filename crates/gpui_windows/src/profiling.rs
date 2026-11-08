use std::{
    marker::PhantomData,
    panic::Location,
    rc::Rc,
    sync::{
        RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

/// The target is an opaque native window key (HWND for the D3D12 backend).
/// The callbacks run on the drawing thread and must not invoke UI or JS code.
#[derive(Clone, Copy)]
pub struct FrameProfilerCallbacks {
    pub begin: fn(usize, &'static str, &'static str, &'static str, u32, u32) -> u64,
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
pub(crate) struct Span {
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
pub(crate) fn span(target: usize, category: &'static str, name: &'static str) -> Span {
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
