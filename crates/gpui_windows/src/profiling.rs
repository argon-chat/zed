// Keep the existing Windows API while sharing the optional hooks with GPUI.
pub use gpui::native_profiling::{FrameProfilerCallbacks, set_frame_profiler};
#[cfg(feature = "native-dx12")]
pub(crate) use gpui::native_profiling::span;
