//! vue-native fork: idle frame pacing.
//!
//! The vsync thread asks every window to paint on every refresh. That keeps a
//! frame clock running for embedders that do work in each frame, and costs a
//! UI-thread wakeup per refresh while nothing changes. An embedder that knows
//! it is idle calls [`pace_frames`] with the time it next needs a frame; until
//! then a refresh is delivered only when something asks for one:
//!
//! - gpui frame demand (a window became dirty, `on_next_frame`, a present
//!   request), through the window's frame waker;
//! - a message to a window (input, focus, resize and the like);
//! - a touchpad gesture or its inertia, which Direct Manipulation advances
//!   only from paints;
//! - [`wake_frames`], from any thread.
//!
//! A delivered refresh ends the pacing: the embedder decides again in that
//! frame. Without a [`pace_frames`] call nothing changes.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Nanoseconds after [`base`] until which refreshes wait for demand; 0 when
/// every refresh is delivered.
static PACED_UNTIL: AtomicU64 = AtomicU64::new(0);
/// Something asked for a frame since the last delivered refresh.
static DEMAND: AtomicBool = AtomicBool::new(false);
/// A touchpad gesture is running or coasting.
static GESTURE: AtomicBool = AtomicBool::new(false);

fn base() -> Instant {
    static BASE: OnceLock<Instant> = OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

fn stamp(at: Instant) -> u64 {
    // Never 0, which means "not paced".
    (at.saturating_duration_since(base()).as_nanos() as u64).max(1)
}

/// Deliver refreshes only on demand until `until`, or every refresh again
/// with `None`.
pub fn pace_frames(until: Option<Instant>) {
    PACED_UNTIL.store(until.map_or(0, stamp), Ordering::Release);
}

/// [`pace_frames`] from inside a frame, before its draw: the demand raised so
/// far is what this frame is about to draw (an embedder that repaints from its
/// frame callback raises it through the window's frame waker), so it does not
/// ask for the next refresh. Demand that arrives later still does; one that
/// raced in just before this call waits for the deadline at most.
pub fn pace_frames_from_frame(until: Instant) {
    DEMAND.store(false, Ordering::Release);
    pace_frames(Some(until));
}

/// Ask for the next refresh to be delivered. Callable from any thread.
pub fn wake_frames() {
    DEMAND.store(true, Ordering::Release);
}

pub(crate) fn set_gesture_active(active: bool) {
    GESTURE.store(active, Ordering::Release);
}

/// Whether a window message asks for a frame: everything but the paint and
/// timer traffic a window gets while idle.
pub(crate) fn message_wants_frame(msg: u32) -> bool {
    use windows::Win32::UI::WindowsAndMessaging::{WM_GETOBJECT, WM_PAINT, WM_TIMER};
    !matches!(msg, WM_PAINT | WM_TIMER | WM_GETOBJECT)
}

/// Called by the vsync thread once per refresh: whether to deliver it.
pub(crate) fn take_due(now: Instant) -> bool {
    let demand = DEMAND.swap(false, Ordering::AcqRel);
    let until = PACED_UNTIL.load(Ordering::Acquire);
    if until == 0 {
        return true;
    }
    if demand || GESTURE.load(Ordering::Acquire) || stamp(now) >= until {
        // Only this pacing: a newer one set meanwhile stands.
        let _ = PACED_UNTIL.compare_exchange(until, 0, Ordering::AcqRel, Ordering::Acquire);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // One test: the state is process-global.
    #[test]
    fn paced_refreshes_wait_for_demand_or_the_deadline() {
        let now = Instant::now();
        pace_frames(None);
        assert!(take_due(now), "unpaced, every refresh is delivered");

        pace_frames(Some(now + Duration::from_millis(30)));
        assert!(!take_due(now));
        assert!(!take_due(now + Duration::from_millis(10)));
        wake_frames();
        assert!(take_due(now + Duration::from_millis(11)), "demand is delivered");
        assert!(take_due(now + Duration::from_millis(12)), "and ends the pacing");

        pace_frames(Some(now + Duration::from_millis(30)));
        assert!(!take_due(now + Duration::from_millis(29)));
        assert!(take_due(now + Duration::from_millis(30)), "the deadline is delivered");

        pace_frames(Some(now + Duration::from_secs(1)));
        set_gesture_active(true);
        assert!(take_due(now), "a gesture keeps refreshes coming");
        set_gesture_active(false);
        pace_frames(Some(now + Duration::from_secs(1)));
        assert!(!take_due(now));

        // Demand raised before the pacing began is still served.
        wake_frames();
        pace_frames(Some(now + Duration::from_secs(1)));
        assert!(take_due(now));

        // Paced from inside a frame, the frame's own demand is what it draws.
        wake_frames();
        pace_frames_from_frame(now + Duration::from_secs(1));
        assert!(!take_due(now));
        wake_frames();
        assert!(take_due(now), "demand after it is served");
        pace_frames(None);
    }

    #[test]
    fn idle_traffic_does_not_ask_for_frames() {
        use windows::Win32::UI::WindowsAndMessaging::*;
        for msg in [WM_PAINT, WM_TIMER, WM_GETOBJECT] {
            assert!(!message_wants_frame(msg));
        }
        for msg in [WM_MOUSEMOVE, WM_KEYDOWN, WM_SIZE, WM_SETFOCUS, WM_MOUSEWHEEL] {
            assert!(message_wants_frame(msg));
        }
    }
}
