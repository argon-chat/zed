//! vue-native's Apache-2.0 replacement for zed's `ztracing` profiling facade.
//!
//! Upstream `ztracing` is licensed GPL-3.0-or-later, and in its default
//! configuration (no `--cfg ztracing`, not wasm) it compiles down to a no-op
//! facade anyway. This shim reimplements that no-op surface so no GPL code
//! enters the dependency graph of a vue-native binary. It was written against
//! the public API only; no upstream code was copied.

pub use tracing::{Level, field};
pub use ztracing_macro::instrument;

/// Do-nothing stand-in for `tracing::Span`.
pub struct Span;

impl Span {
    pub fn current() -> Self {
        Span
    }

    pub fn enter(&self) {}

    pub fn record<K, V>(&self, _key: K, _value: V) {}
}

/// Every span/event macro discards its tokens and yields [`Span`].
#[macro_export]
macro_rules! __ztracing_noop {
    ($($tokens:tt)*) => {
        $crate::Span
    };
}

pub use __ztracing_noop as debug_span;
pub use __ztracing_noop as error_span;
pub use __ztracing_noop as event;
pub use __ztracing_noop as info_span;
pub use __ztracing_noop as span;
pub use __ztracing_noop as trace_span;
pub use __ztracing_noop as warn_span;

pub fn init() {}
