//! vue-native fork: pointer lock. The cursor is hidden, clipped to the client area and put back
//! at its centre after every move, so motion never stops at a screen edge. Moves report the
//! cursor's (accelerated) displacement from the centre; unadjusted moves report raw input
//! (`WM_INPUT`) counts instead.

use std::cell::Cell;

use gpui::{Pixels, Point, PointerLockError, point, px};
use windows::Win32::{
    Foundation::*,
    Graphics::Gdi::ClientToScreen,
    UI::{Input::*, WindowsAndMessaging::*},
};

#[derive(Clone, Copy)]
struct Held {
    hwnd: HWND,
    unadjusted: bool,
    /// Screen pixels: where the cursor goes back to on release.
    restore: POINT,
    /// Screen pixels: where the cursor is held.
    center: POINT,
    /// Raw input came with absolute positions (a remote session, a pen): measure the cursor instead.
    raw_absolute: bool,
}

#[derive(Default)]
pub(crate) struct PointerLock {
    held: Cell<Option<Held>>,
    motion: Cell<Option<Point<Pixels>>>,
}

impl PointerLock {
    pub(crate) fn lock(&self, hwnd: HWND, unadjusted: bool) -> Result<(), PointerLockError> {
        self.unlock();
        if unsafe { GetForegroundWindow() } != hwnd {
            return Err(PointerLockError::Refused);
        }
        let mut restore = POINT::default();
        unsafe { GetCursorPos(&mut restore) }.map_err(|_| PointerLockError::Refused)?;
        if unadjusted && register_raw_mouse(Some(hwnd)).is_err() {
            return Err(PointerLockError::Refused);
        }
        let Some(center) = hold(hwnd) else {
            if unadjusted {
                register_raw_mouse(None).ok();
            }
            return Err(PointerLockError::Refused);
        };
        self.held.set(Some(Held {
            hwnd,
            unadjusted,
            restore,
            center,
            raw_absolute: false,
        }));
        self.motion.set(None);
        unsafe { SetCursor(None) };
        Ok(())
    }

    pub(crate) fn unlock(&self) {
        let Some(held) = self.held.take() else {
            return;
        };
        self.motion.set(None);
        unsafe {
            ClipCursor(None).ok();
            if held.unadjusted {
                register_raw_mouse(None).ok();
            }
            SetCursorPos(held.restore.x, held.restore.y).ok();
        }
    }

    pub(crate) fn is_locked(&self) -> bool {
        self.held.get().is_some()
    }

    pub(crate) fn take_motion(&self) -> Option<Point<Pixels>> {
        self.motion.take()
    }

    /// The client area moved or changed size: clip to where it is now.
    pub(crate) fn follow_window(&self) {
        let Some(mut held) = self.held.get() else {
            return;
        };
        match hold(held.hwnd) {
            Some(center) => {
                held.center = center;
                self.held.set(Some(held));
            }
            None => self.unlock(),
        }
    }

    /// A `WM_MOUSEMOVE`: `None` when not locked, otherwise whether it carried motion to report.
    /// The cursor is read now rather than from the message, so no motion since is lost to the
    /// recentring.
    pub(crate) fn on_mouse_move(&self, scale: f32) -> Option<bool> {
        let held = self.held.get()?;
        let mut at = POINT::default();
        if unsafe { GetCursorPos(&mut at) }.is_err() {
            return Some(false);
        }
        let (dx, dy) = (at.x - held.center.x, at.y - held.center.y);
        if dx == 0 && dy == 0 {
            return Some(false);
        }
        unsafe { SetCursorPos(held.center.x, held.center.y) }.ok();
        if held.unadjusted && !held.raw_absolute {
            return Some(false);
        }
        self.add_motion(point(px(dx as f32 / scale), px(dy as f32 / scale)));
        Some(true)
    }

    /// A `WM_INPUT`: whether it carried unadjusted motion to report.
    pub(crate) fn on_raw_input(&self, lparam: LPARAM) -> bool {
        let Some(mut held) = self.held.get().filter(|held| held.unadjusted) else {
            return false;
        };
        let mut raw = RAWINPUT::default();
        let mut size = std::mem::size_of::<RAWINPUT>() as u32;
        // SAFETY: `raw` is a RAWINPUT of `size` bytes; a mouse record fits in one.
        let read = unsafe {
            GetRawInputData(
                HRAWINPUT(lparam.0 as _),
                RID_INPUT,
                Some(&mut raw as *mut RAWINPUT as *mut _),
                &mut size,
                std::mem::size_of::<RAWINPUTHEADER>() as u32,
            )
        };
        if read == u32::MAX || raw.header.dwType != RIM_TYPEMOUSE.0 {
            return false;
        }
        // SAFETY: the header says this record is a mouse's.
        let mouse = unsafe { raw.data.mouse };
        if mouse.usFlags.0 & MOUSE_MOVE_ABSOLUTE.0 != 0 {
            held.raw_absolute = true;
            self.held.set(Some(held));
            return false;
        }
        if mouse.lLastX == 0 && mouse.lLastY == 0 {
            return false;
        }
        self.add_motion(point(px(mouse.lLastX as f32), px(mouse.lLastY as f32)));
        true
    }

    fn add_motion(&self, by: Point<Pixels>) {
        let sum = self.motion.get().map_or(by, |m| m + by);
        self.motion.set(Some(sum));
    }
}

/// Clips the cursor to the client area and puts it at the centre; `None` when there is no area
/// (minimized) or the system refused.
fn hold(hwnd: HWND) -> Option<POINT> {
    let mut rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut rect) }.ok()?;
    let mut top_left = POINT {
        x: rect.left,
        y: rect.top,
    };
    let mut bottom_right = POINT {
        x: rect.right,
        y: rect.bottom,
    };
    unsafe {
        if !ClientToScreen(hwnd, &mut top_left).as_bool()
            || !ClientToScreen(hwnd, &mut bottom_right).as_bool()
        {
            return None;
        }
    }
    if bottom_right.x <= top_left.x || bottom_right.y <= top_left.y {
        return None;
    }
    let clip = RECT {
        left: top_left.x,
        top: top_left.y,
        right: bottom_right.x,
        bottom: bottom_right.y,
    };
    unsafe { ClipCursor(Some(&clip)) }.ok()?;
    let center = POINT {
        x: (top_left.x + bottom_right.x) / 2,
        y: (top_left.y + bottom_right.y) / 2,
    };
    unsafe { SetCursorPos(center.x, center.y) }.ok()?;
    Some(center)
}

/// Raw mouse input for `hwnd` while it is the foreground window; `None` stops it.
fn register_raw_mouse(hwnd: Option<HWND>) -> windows::core::Result<()> {
    let device = RAWINPUTDEVICE {
        usUsagePage: 0x01,
        usUsage: 0x02,
        dwFlags: if hwnd.is_some() {
            RAWINPUTDEVICE_FLAGS(0)
        } else {
            RIDEV_REMOVE
        },
        hwndTarget: hwnd.unwrap_or_default(),
    };
    unsafe { RegisterRawInputDevices(&[device], std::mem::size_of::<RAWINPUTDEVICE>() as u32) }
}
