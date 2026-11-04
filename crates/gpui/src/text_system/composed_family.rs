//! vue-native fork: font families made of several faces, each of which draws
//! only the characters of its own `unicode-range`.
//!
//! A web font is often shipped as one CSS family cut into subsets (Fontsource
//! and Google Fonts: `latin`, `latin-ext`, `cyrillic`, …), one file and one
//! `@font-face` rule each. A browser draws every character with the face of
//! the family whose `unicode-range` covers it (CSS Fonts 4 §4.5 and §5.3), and
//! a character no face covers with the next family in the list. The platform
//! text system matches faces by family name, weight and style only, so each
//! face is registered under a family of its own and the family the page names
//! is a [`FamilyComposer`]: which faces a run of that family is drawn with, for
//! the run's weight and style, and which characters each one draws. The
//! platform splits the run by character before shaping.

use std::sync::Arc;

use crate::{FontStyle, FontWeight, SharedString};

/// One face of a composed family: the family it is registered under (a family
/// of its own, holding that one font file) and the characters it draws.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComposedFace {
    /// The family the face's file was registered under.
    pub family: SharedString,
    /// Inclusive code point ranges; empty means every code point.
    pub ranges: Arc<[(u32, u32)]>,
}

impl ComposedFace {
    /// Does the face draw `c`?
    pub fn covers(&self, c: char) -> bool {
        let c = c as u32;
        self.ranges.is_empty() || self.ranges.iter().any(|&(start, end)| start <= c && c <= end)
    }
}

/// The faces a composed family draws text of a weight and style with, in the
/// order they are tried for a character: the first that covers it draws it.
/// Empty when the family has no face for that text (yet).
pub type FamilyComposer = Arc<dyn Fn(FontWeight, FontStyle) -> Vec<ComposedFace> + Send + Sync>;

/// Is `c` a character that belongs to the cluster before it, so that it is
/// drawn with the same face as that character rather than chosen on its own?
/// Combining marks, joiners and variation selectors: a browser keeps a
/// grapheme in one font when it can.
pub fn continues_cluster(c: char) -> bool {
    matches!(
        c as u32,
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x200C..=0x200D
            | 0x20D0..=0x20FF
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
            | 0xE0100..=0xE01EF
    )
}

/// Splits `text` into the pieces each face of `faces` draws: `(byte length,
/// index into faces)`, `None` for a piece no face covers. Adjacent characters
/// with the same answer are one piece; a [`continues_cluster`] character
/// stays with the piece before it.
pub fn split_by_face(text: &str, faces: &[ComposedFace]) -> Vec<(usize, Option<usize>)> {
    let mut out: Vec<(usize, Option<usize>)> = Vec::new();
    for c in text.chars() {
        let len = c.len_utf8();
        let face = match out.last() {
            Some(&(_, prev)) if continues_cluster(c) => prev,
            _ => faces.iter().position(|f| f.covers(c)),
        };
        match out.last_mut() {
            Some(last) if last.1 == face => last.0 += len,
            _ => out.push((len, face)),
        }
    }
    out
}

// Tested in gpui_windows (`direct_write::composed_tests`), whose tests run in
// vue-native's workspace; this crate's need dev-dependencies it does not have.
