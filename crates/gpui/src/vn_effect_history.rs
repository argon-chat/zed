//! vue-native: the state of `[SkHistory]` effects, which a renderer keeps from
//! frame to frame per [`crate::VnEffectDraw::history`] key: two textures the
//! size of the drawn quad, one holding the last state while the other takes
//! the next.

use std::collections::HashMap;

/// Bytes per history texel: RGBA, each a 16-bit float.
pub const VN_HISTORY_TEXEL_BYTES: u64 = 8;
/// A quad longer than this on a side, in device pixels, gets no history.
pub const VN_HISTORY_MAX_SIDE: u32 = 4096;
/// What the histories of one window may hold together, both textures counted.
pub const VN_HISTORY_BUDGET_BYTES: u64 = 256 << 20;
/// The shortest time between two steps of one history, so a simulation runs
/// at about the same speed on a 60 Hz and a 144 Hz display.
pub const VN_HISTORY_MIN_STEP_SECONDS: f32 = 1. / 75.;

/// One renderer's histories; `T` is its texture handle.
pub struct VnEffectHistories<T> {
    entries: HashMap<u64, Entry<T>>,
    bytes: u64,
}

struct Entry<T> {
    size: (u32, u32),
    textures: [T; 2],
    /// The texture holding the last state.
    read: usize,
    last_step: f32,
    used: bool,
    stepped: bool,
}

/// One history in this frame.
pub struct VnHistory<T> {
    /// The state the effect reads, as the last step left it.
    pub read: T,
    /// The texture this frame's step writes, when the history steps.
    pub write: Option<T>,
    /// Both textures were just made and must be cleared to zero first.
    pub fresh: bool,
}

impl<T> Default for VnEffectHistories<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
        }
    }
}

impl<T: Clone> VnEffectHistories<T> {
    /// The history for `key` at `size`, made with `create` when it is new or
    /// its size changed (which starts it over). None for key 0, a quad too
    /// large, a full budget, or a texture `create` could not make.
    pub fn prepare(
        &mut self,
        key: u64,
        size: (u32, u32),
        now: f32,
        mut create: impl FnMut(u32, u32) -> Option<T>,
    ) -> Option<VnHistory<T>> {
        let (width, height) = size;
        if key == 0 || width == 0 || height == 0 {
            return None;
        }
        if width > VN_HISTORY_MAX_SIDE || height > VN_HISTORY_MAX_SIDE {
            advisory(format!(
                "an `[SkHistory]` effect on an element larger than {VN_HISTORY_MAX_SIDE} device \
                 pixels on a side keeps no history; it reads zeros there"
            ));
            return None;
        }
        if self
            .entries
            .get(&key)
            .is_some_and(|entry| entry.size != size)
        {
            self.remove(key);
        }
        let mut fresh = false;
        if !self.entries.contains_key(&key) {
            let cost = 2 * u64::from(width) * u64::from(height) * VN_HISTORY_TEXEL_BYTES;
            if self.bytes + cost > VN_HISTORY_BUDGET_BYTES {
                advisory(format!(
                    "`[SkHistory]` effects in one window are limited to {} MiB of history; \
                     the ones past it read zeros",
                    VN_HISTORY_BUDGET_BYTES >> 20
                ));
                return None;
            }
            let textures = [create(width, height)?, create(width, height)?];
            self.entries.insert(
                key,
                Entry {
                    size,
                    textures,
                    read: 0,
                    last_step: f32::NEG_INFINITY,
                    used: false,
                    stepped: false,
                },
            );
            self.bytes += cost;
            fresh = true;
        }
        let entry = self.entries.get_mut(&key)?;
        entry.used = true;
        let step = !entry.stepped
            && (fresh
                || now - entry.last_step >= VN_HISTORY_MIN_STEP_SECONDS
                || now < entry.last_step);
        if step {
            entry.stepped = true;
            entry.last_step = now;
        }
        Some(VnHistory {
            read: entry.textures[entry.read].clone(),
            write: step.then(|| entry.textures[1 - entry.read].clone()),
            fresh,
        })
    }

    /// Ends a frame: a history that stepped reads its new state from now on,
    /// and one the frame did not draw is freed.
    pub fn finish_frame(&mut self) {
        let mut freed = 0;
        self.entries.retain(|_, entry| {
            if !entry.used {
                freed +=
                    2 * u64::from(entry.size.0) * u64::from(entry.size.1) * VN_HISTORY_TEXEL_BYTES;
                return false;
            }
            if entry.stepped {
                entry.read = 1 - entry.read;
            }
            entry.used = false;
            entry.stepped = false;
            true
        });
        self.bytes -= freed;
    }

    /// Drops every history (a lost device).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    /// The bytes the histories hold.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// How many histories are held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether none is held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn remove(&mut self, key: u64) {
        if let Some(entry) = self.entries.remove(&key) {
            self.bytes -=
                2 * u64::from(entry.size.0) * u64::from(entry.size.1) * VN_HISTORY_TEXEL_BYTES;
        }
    }
}

fn advisory(message: String) {
    use parking_lot::Mutex;
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock();
    if seen.contains(&message) {
        return;
    }
    log::warn!("{message}");
    eprintln!("skathi: {message}");
    seen.push(message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_history_ping_pongs_steps_at_most_at_the_step_rate_and_frees_when_not_drawn() {
        let mut made = 0;
        let mut histories = VnEffectHistories::default();
        let mut create = |_, _| {
            made += 1;
            Some(made)
        };
        let first = histories.prepare(7, (4, 4), 1.0, &mut create).unwrap();
        assert!(first.fresh);
        assert_eq!((first.read, first.write), (1, Some(2)));
        histories.finish_frame();

        let soon = histories.prepare(7, (4, 4), 1.001, &mut create).unwrap();
        assert_eq!(
            (soon.read, soon.write),
            (2, None),
            "reads the step it wrote"
        );
        histories.finish_frame();

        let later = histories.prepare(7, (4, 4), 1.1, &mut create).unwrap();
        assert_eq!((later.read, later.write), (2, Some(1)));
        histories.finish_frame();

        let resized = histories.prepare(7, (5, 4), 1.2, &mut create).unwrap();
        assert!(resized.fresh, "a new size starts over");
        assert_eq!(histories.bytes(), 2 * 5 * 4 * VN_HISTORY_TEXEL_BYTES);
        histories.finish_frame();

        histories.finish_frame();
        assert!(histories.is_empty(), "a frame without it frees it");
        assert_eq!(histories.bytes(), 0);
    }

    #[test]
    fn a_history_past_the_budget_or_the_side_limit_is_refused() {
        let mut histories = VnEffectHistories::default();
        let side = VN_HISTORY_MAX_SIDE;
        assert!(
            histories
                .prepare(1, (side + 1, 1), 0., |_, _| Some(()))
                .is_none()
        );
        assert!(
            histories
                .prepare(1, (side, side), 0., |_, _| Some(()))
                .is_some()
        );
        assert!(
            histories.prepare(2, (1, 1), 0., |_, _| Some(())).is_none(),
            "a 4096² history fills the budget"
        );
        assert!(histories.prepare(0, (1, 1), 0., |_, _| Some(())).is_none());
    }
}
