use crate::{
    FontId, GlyphId, Hyphens, Pixels, PlatformTextSystem, Point, SharedString, Size, TextWrap,
    WordBreak, point, px,
};
use collections::{FxHashMap, FxHashSet};
use icu_segmenter::{
    LineSegmenter,
    options::{LineBreakOptions, LineBreakWordOption},
};
use itertools::Itertools;
use smallvec::SmallVec;
use std::{
    borrow::Borrow,
    cell::RefCell,
    hash::{Hash, Hasher},
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use super::LineWrapper;

/// vue-native fork: line lookups and the shaping they missed into, process-wide.
/// One relaxed add per lookup and a clock pair per shape, so an embedder's
/// frame trace can say how much of a frame went to text shaping.
static LINE_LOOKUPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LINE_SHAPES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LINE_SHAPE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// vue-native fork: `(lookups, shapes, nanoseconds spent shaping)` since the
/// process started, over every window's line layout cache. A lookup that is
/// not a shape was answered from this frame's or the previous frame's cache.
pub fn line_layout_counters() -> (u64, u64, u64) {
    (
        LINE_LOOKUPS.load(Ordering::Relaxed),
        LINE_SHAPES.load(Ordering::Relaxed),
        LINE_SHAPE_NANOS.load(Ordering::Relaxed),
    )
}

fn count_lookup() {
    LINE_LOOKUPS.fetch_add(1, Ordering::Relaxed);
}

fn count_shape(started: std::time::Instant) {
    LINE_SHAPES.fetch_add(1, Ordering::Relaxed);
    LINE_SHAPE_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// A laid out and styled line of text
#[derive(Default, Debug)]
pub struct LineLayout {
    /// The font size for this line
    pub font_size: Pixels,
    /// The width of the line
    pub width: Pixels,
    /// The ascent of the line
    pub ascent: Pixels,
    /// The descent of the line
    pub descent: Pixels,
    /// The shaped runs that make up this line
    pub runs: Vec<ShapedRun>,
    /// The length of the line in utf-8 bytes
    pub len: usize,
}

/// A run of text that has been shaped .
#[derive(Debug, Clone)]
pub struct ShapedRun {
    /// The font id for this run
    pub font_id: FontId,
    /// The glyphs that make up this run
    pub glyphs: Vec<ShapedGlyph>,
}

/// A single glyph, ready to paint.
#[derive(Clone, Debug)]
pub struct ShapedGlyph {
    /// The ID for this glyph, as determined by the text system.
    pub id: GlyphId,

    /// The position of this glyph in its containing line.
    pub position: Point<Pixels>,

    /// The index of this glyph in the original text.
    pub index: usize,

    /// Whether this glyph is an emoji
    pub is_emoji: bool,
}

impl LineLayout {
    /// The index for the character at the given x coordinate
    pub fn index_for_x(&self, x: Pixels) -> Option<usize> {
        if x >= self.width {
            None
        } else {
            for run in self.runs.iter().rev() {
                for glyph in run.glyphs.iter().rev() {
                    if glyph.position.x <= x {
                        return Some(glyph.index);
                    }
                }
            }
            Some(0)
        }
    }

    /// closest_index_for_x returns the character boundary closest to the given x coordinate
    /// (e.g. to handle aligning up/down arrow keys)
    pub fn closest_index_for_x(&self, x: Pixels) -> usize {
        let mut prev_index = 0;
        let mut prev_x = px(0.);

        for run in self.runs.iter() {
            for glyph in run.glyphs.iter() {
                if glyph.position.x >= x {
                    if glyph.position.x - x < x - prev_x {
                        return glyph.index;
                    } else {
                        return prev_index;
                    }
                }
                prev_index = glyph.index;
                prev_x = glyph.position.x;
            }
        }

        if self.len == 1 {
            if x > self.width / 2. {
                return 1;
            } else {
                return 0;
            }
        }

        self.len
    }

    /// The x position of the character at the given index
    pub fn x_for_index(&self, index: usize) -> Pixels {
        for run in &self.runs {
            for glyph in &run.glyphs {
                if glyph.index >= index {
                    return glyph.position.x;
                }
            }
        }
        self.width
    }

    /// The corresponding Font at the given index
    pub fn font_id_for_index(&self, index: usize) -> Option<FontId> {
        for run in &self.runs {
            for glyph in &run.glyphs {
                if glyph.index >= index {
                    return Some(run.font_id);
                }
            }
        }
        None
    }

    /// Split this layout at a byte index, returning `(prefix, suffix)`.
    ///
    /// - `prefix` contains glyphs for bytes `[0, byte_index)` with original positions.
    ///   Its width equals the x-advance up to the split point.
    /// - `suffix` contains glyphs for bytes `[byte_index, len)` with positions
    ///   shifted left so the first glyph starts at x=0, and byte indices rebased to 0.
    /// - `font_size`, `ascent`, and `descent` are copied to both halves.
    pub fn split_at(&self, byte_index: usize) -> (LineLayout, LineLayout) {
        let x_offset = self.x_for_index(byte_index);

        // Partition glyph runs. A single run may contribute glyphs to both halves.
        let mut left_runs = Vec::new();
        let mut right_runs = Vec::new();

        for run in &self.runs {
            let split_pos = run.glyphs.partition_point(|g| g.index < byte_index);

            if split_pos > 0 {
                left_runs.push(ShapedRun {
                    font_id: run.font_id,
                    glyphs: run.glyphs[..split_pos].to_vec(),
                });
            }

            if split_pos < run.glyphs.len() {
                let right_glyphs = run.glyphs[split_pos..]
                    .iter()
                    .map(|g| ShapedGlyph {
                        id: g.id,
                        position: point(g.position.x - x_offset, g.position.y),
                        index: g.index - byte_index,
                        is_emoji: g.is_emoji,
                    })
                    .collect();
                right_runs.push(ShapedRun {
                    font_id: run.font_id,
                    glyphs: right_glyphs,
                });
            }
        }

        let left = LineLayout {
            font_size: self.font_size,
            width: x_offset,
            ascent: self.ascent,
            descent: self.descent,
            runs: left_runs,
            len: byte_index,
        };

        let right = LineLayout {
            font_size: self.font_size,
            width: self.width - x_offset,
            ascent: self.ascent,
            descent: self.descent,
            runs: right_runs,
            len: self.len - byte_index,
        };

        (left, right)
    }

    /// vue-native fork: where the line breaks when wrapped at `wrap_width`
    /// under `rules` (CSS Text 3 §5). A line breaks at its last soft wrap
    /// opportunity that fits ([`LineLayout::wrap_glyphs`]). A word that fits
    /// nowhere is broken where it overflows when `rules` allow that
    /// (`overflow-wrap: break-word | anywhere`, `word-break: break-word`), and
    /// otherwise overflows its line, which then breaks at the next opportunity.
    /// `hyphen_width` is the width of the hyphen a soft-hyphen break shows: a
    /// line that ends in one must fit it.
    pub(crate) fn wrap(
        &self,
        text: &str,
        wrap_width: Pixels,
        max_lines: Option<usize>,
        rules: TextWrap,
        hyphen_width: Pixels,
        scratch: &mut WrapScratch,
    ) -> Wrapping {
        let mut out = Wrapping::default();
        // The last opportunity on the current line: the glyph the next line
        // would start at, its x, where the line's content ends before it
        // (trailing spaces hang; a hyphen counts), and whether it shows a
        // hyphen.
        let mut candidate: Option<(WrapBoundary, Pixels, Pixels, bool)> = None;
        let mut last_boundary = WrapBoundary {
            run_ix: 0,
            glyph_ix: 0,
        };
        let mut last_boundary_x = px(0.);
        // The end of the current line's last glyph that is not a space.
        let mut content_end = px(0.);
        // The current line holds a word that may not break and is past the
        // edge: it breaks at the next opportunity, wherever that is.
        let mut overflowing = false;
        let mut widest = px(0.);
        let full = |out: &Wrapping| {
            max_lines.is_some_and(|max| out.boundaries.len() >= max.saturating_sub(1))
        };

        'glyphs: for WrapGlyph {
            boundary,
            prev_ch,
            ch,
            x,
            next_x,
            opportunity,
        } in self.wrap_glyphs(text, rules, scratch)
        {
            if let Some(hyphen) = opportunity {
                let end = if hyphen {
                    content_end + hyphen_width
                } else {
                    content_end
                };
                if overflowing && boundary > last_boundary {
                    // The overflowing word ends here: so does its line.
                    if full(&out) {
                        break 'glyphs;
                    }
                    widest = widest.max(end - last_boundary_x);
                    out.push(boundary, hyphen);
                    last_boundary = boundary;
                    last_boundary_x = x;
                    overflowing = false;
                    candidate = None;
                } else if !hyphen || end - last_boundary_x <= wrap_width {
                    candidate = Some((boundary, x, end, hyphen));
                }
            }

            // A space that crosses the edge hangs (CSS Text 3 §4.1.3): the
            // line breaks before the next word, not before the space, which
            // would start the next line with it.
            while !overflowing
                && next_x - last_boundary_x > wrap_width
                && boundary > last_boundary
                && ch != ' '
            {
                // When used line_clamp, we should limit the number of lines.
                if full(&out) {
                    break 'glyphs;
                }
                if let Some((at, at_x, end, hyphen)) = candidate.take() {
                    widest = widest.max(end - last_boundary_x);
                    out.push(at, hyphen);
                    last_boundary = at;
                    last_boundary_x = at_x;
                    // The rest of the word may still not fit on the new
                    // line: looked at again.
                } else if rules.breaks_words() && !continues_cluster(prev_ch, ch) {
                    widest = widest.max(x - last_boundary_x);
                    out.push(boundary, false);
                    last_boundary = boundary;
                    last_boundary_x = x;
                } else {
                    overflowing = true;
                }
            }
            if ch != ' ' {
                content_end = next_x;
            }
        }
        widest = widest.max(content_end - last_boundary_x);
        if widest > wrap_width {
            out.overflow = widest;
        }
        out
    }

    /// The line's min-content width (CSS Sizing 3 §5.1): its widest stretch
    /// between two of the soft wrap opportunities [`Self::wrap`] breaks at,
    /// without the spaces that end it, which hang. Under `overflow-wrap:
    /// anywhere` (and `word-break: break-word`) a break inside a word counts
    /// too, so the widest stretch is one character. Wrapped at this width,
    /// the line breaks at every opportunity it counted.
    pub(crate) fn min_content_width(
        &self,
        text: &str,
        rules: TextWrap,
        hyphen_width: Pixels,
    ) -> Pixels {
        let mut widest = px(0.);
        let mut start_x = px(0.);
        let mut end_x = px(0.);
        let mut seen_non_whitespace = false;
        let mut scratch = WrapScratch::default();
        for WrapGlyph {
            prev_ch,
            ch,
            x,
            next_x,
            opportunity,
            ..
        } in self.wrap_glyphs(text, rules, &mut scratch)
        {
            let opportunity = opportunity.or_else(|| {
                (rules.min_content_breaks_words()
                    && seen_non_whitespace
                    && ch != ' '
                    && !continues_cluster(prev_ch, ch))
                .then_some(false)
            });
            if let Some(hyphen) = opportunity {
                let hyphen = if hyphen { hyphen_width } else { px(0.) };
                widest = widest.max(end_x - start_x + hyphen);
                start_x = x;
            }
            if ch != ' ' {
                seen_non_whitespace = true;
                end_x = next_x;
            }
        }
        widest.max(end_x - start_x)
    }

    /// vue-native fork: the line's glyphs in visual order, one per shaped
    /// cluster, newlines skipped, each with the soft wrap opportunity before
    /// it under `rules`.
    ///
    /// The opportunities are UAX #14's ([`css_break_indices`], tailored by
    /// `word-break` and `hyphens`), less those inside a shaped cluster.
    /// Transitions between line-break segments are detected rather than
    /// glyph indices matched directly, to accommodate RTL text (a workaround
    /// for the lack of complete UAX #9 support). The first segment that is
    /// not a space opens no opportunity: a line never breaks before its first
    /// word.
    fn wrap_glyphs<'a>(
        &'a self,
        text: &'a str,
        rules: TextWrap,
        scratch: &'a mut WrapScratch,
    ) -> impl Iterator<Item = WrapGlyph> + 'a {
        // Shaping can combine characters across a line break opportunity, so we ignore those
        // break opportunities that would separate them.
        let WrapScratch {
            glyph_indices,
            break_indices,
        } = scratch;
        glyph_indices.clear();
        glyph_indices.extend(
            self.runs
                .iter()
                .flat_map(|run| run.glyphs.iter().map(|glyph| glyph.index)),
        );
        break_indices.clear();
        break_indices
            .extend(css_break_indices(text, rules).filter(|index| glyph_indices.contains(index)));
        let break_indices = &*break_indices;
        let mut glyphs = self
            .runs
            .iter()
            .enumerate()
            .flat_map(move |(run_ix, run)| {
                run.glyphs.iter().enumerate().map(move |(glyph_ix, glyph)| {
                    (
                        WrapBoundary { run_ix, glyph_ix },
                        glyph.index,
                        glyph.position.x,
                    )
                })
            })
            .dedup_by(|(_, previous_index, ..), (_, index, ..)| previous_index == index)
            .peekable();
        let mut current_segment: Option<Range<usize>> = None;
        let mut prev_ch = '\0';
        std::iter::from_fn(move || {
            loop {
                let (boundary, index, x) = glyphs.next()?;
                let next_x = glyphs.peek().map_or(self.width, |(_, _, x)| *x);
                let ch = text[index..].chars().next().unwrap_or(' ');
                if ch == '\n' {
                    continue;
                }
                let mut opportunity = None;
                if ch != ' '
                    && current_segment
                        .as_ref()
                        .is_none_or(|segment| !segment.contains(&index))
                {
                    let segment =
                        break_indices.partition_point(|break_index| *break_index <= index);
                    let start = segment
                        .checked_sub(1)
                        .and_then(|previous| break_indices.get(previous))
                        .copied()
                        .unwrap_or(0);
                    let end = break_indices.get(segment).copied().unwrap_or(text.len());
                    if current_segment.is_some() {
                        // A break after a soft hyphen shows a hyphen; under
                        // `hyphens: none` there is no such break.
                        opportunity = Some(text[..start].ends_with('\u{AD}'));
                    }
                    current_segment = Some(start..end);
                }
                let glyph = WrapGlyph {
                    boundary,
                    prev_ch,
                    ch,
                    x,
                    next_x,
                    opportunity,
                };
                prev_ch = ch;
                return Some(glyph);
            }
        })
    }
}

/// vue-native fork: one glyph of [`LineLayout::wrap_glyphs`].
struct WrapGlyph {
    boundary: WrapBoundary,
    /// The character of the glyph before, `'\0'` for the first.
    prev_ch: char,
    ch: char,
    x: Pixels,
    /// Where the next glyph starts, or the line's width after the last.
    next_x: Pixels,
    /// A soft wrap opportunity before this glyph: `Some(true)` when the
    /// break shows a hyphen.
    opportunity: Option<bool>,
}

/// vue-native fork: what [`LineLayout::wrap`] decided for one line of text.
#[derive(Clone, Debug, Default)]
pub(crate) struct Wrapping {
    pub(crate) boundaries: SmallVec<[WrapBoundary; 1]>,
    /// The boundaries a soft hyphen made, as indices into `boundaries`: the
    /// line before each one ends with a visible hyphen.
    pub(crate) hyphens: SmallVec<[usize; 1]>,
    /// The widest line, trailing spaces not counted, when one is wider than
    /// the wrap width (a word that may not break). Zero otherwise.
    pub(crate) overflow: Pixels,
}

impl Wrapping {
    fn push(&mut self, at: WrapBoundary, hyphen: bool) {
        if hyphen {
            self.hyphens.push(self.boundaries.len());
        }
        self.boundaries.push(at);
    }
}

/// vue-native fork: the byte offsets in `text` a line may break before under
/// `rules`: UAX #14's opportunities ([`LineWrapper::break_indices`]), tailored
/// by CSS Text 3:
/// - `word-break: break-all`: between any two letters, though not before
///   punctuation that never starts a line; `keep-all`: no break between two
///   CJK letters (UAX #14's own tailorings, the same as CSS's);
/// - after a soft hyphen (U+00AD) only under `hyphens: manual`;
/// - after a zero width space (U+200B), as UAX #14 has it.
///
/// No opportunity splits a grapheme cluster.
fn css_break_indices(text: &str, rules: TextWrap) -> impl Iterator<Item = usize> + '_ {
    let word_option = match rules.word_break {
        WordBreak::BreakAll => Some(LineBreakWordOption::BreakAll),
        WordBreak::KeepAll => Some(LineBreakWordOption::KeepAll),
        WordBreak::Normal | WordBreak::BreakWord => None,
    };
    let indices: Box<dyn Iterator<Item = usize> + '_> = match word_option {
        None => Box::new(LineWrapper::break_indices(text)),
        Some(word_option) => {
            let mut options = LineBreakOptions::default();
            options.word_option = Some(word_option);
            Box::new(LineSegmenter::new_for_non_complex_scripts(options).segment_str(text))
        }
    };
    indices.filter(move |index| {
        rules.hyphens == Hyphens::Manual || !text[..*index].ends_with('\u{AD}')
    })
}

/// Does `ch` belong to the grapheme cluster `prev` is in? A combining mark,
/// a joiner, a variation selector, an emoji modifier or tag, or anything
/// after a zero width joiner. A shaper usually gives a cluster one glyph, but
/// not always; a break between two of its glyphs would split a character.
/// Used where a line breaks inside a word (`overflow-wrap`), which UAX #14's
/// opportunities say nothing about.
fn continues_cluster(prev: char, ch: char) -> bool {
    prev == '\u{200D}'
        || matches!(
            ch,
            '\u{0300}'..='\u{036F}'
                | '\u{0483}'..='\u{0489}'
                | '\u{0591}'..='\u{05BD}'
                | '\u{0610}'..='\u{061A}'
                | '\u{064B}'..='\u{065F}'
                | '\u{0900}'..='\u{0903}'
                | '\u{093A}'..='\u{094F}'
                | '\u{1AB0}'..='\u{1AFF}'
                | '\u{1DC0}'..='\u{1DFF}'
                | '\u{200C}'..='\u{200D}'
                | '\u{20D0}'..='\u{20FF}'
                | '\u{FE00}'..='\u{FE0F}'
                | '\u{FE20}'..='\u{FE2F}'
                | '\u{1F3FB}'..='\u{1F3FF}'
                | '\u{E0020}'..='\u{E007F}'
                | '\u{E0100}'..='\u{E01EF}'
        )
}

/// A line of text that has been wrapped to fit a given width
#[derive(Default, Debug)]
pub struct WrappedLineLayout {
    /// The line layout, pre-wrapping.
    pub unwrapped_layout: Arc<LineLayout>,

    /// The boundaries at which the line was wrapped
    pub wrap_boundaries: SmallVec<[WrapBoundary; 1]>,

    /// The width of the line, if it was wrapped
    pub wrap_width: Option<Pixels>,

    /// vue-native fork: the wrap boundaries a soft hyphen made, as indices
    /// into `wrap_boundaries`. The line before each ends with a hyphen.
    pub(crate) hyphen_breaks: SmallVec<[usize; 1]>,

    /// vue-native fork: the widest line, when a word that may not break made
    /// it wider than the wrap width. Zero otherwise.
    pub(crate) overflow_width: Pixels,
}

/// A boundary at which a line was wrapped
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WrapBoundary {
    /// The index in the run just before the line was wrapped
    pub run_ix: usize,
    /// The index of the glyph just before the line was wrapped
    pub glyph_ix: usize,
}

impl WrappedLineLayout {
    /// The length of the underlying text, in utf8 bytes.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.unwrapped_layout.len
    }

    /// The width of this line, in pixels, whether or not it was wrapped.
    pub fn width(&self) -> Pixels {
        self.wrap_width
            .unwrap_or(Pixels::MAX)
            .min(self.unwrapped_layout.width)
            // A line that overflows the wrap width is as wide as it is.
            .max(self.overflow_width)
    }

    /// The size of the whole wrapped text, for the given line_height.
    /// can span multiple lines if there are multiple wrap boundaries.
    pub fn size(&self, line_height: Pixels) -> Size<Pixels> {
        Size {
            width: self.width(),
            height: line_height * (self.wrap_boundaries.len() + 1),
        }
    }

    /// The ascent of a line in this layout
    pub fn ascent(&self) -> Pixels {
        self.unwrapped_layout.ascent
    }

    /// The descent of a line in this layout
    pub fn descent(&self) -> Pixels {
        self.unwrapped_layout.descent
    }

    /// The wrap boundaries in this layout
    pub fn wrap_boundaries(&self) -> &[WrapBoundary] {
        &self.wrap_boundaries
    }

    /// The font size of this layout
    pub fn font_size(&self) -> Pixels {
        self.unwrapped_layout.font_size
    }

    /// The runs in this layout, sans wrapping
    pub fn runs(&self) -> &[ShapedRun] {
        &self.unwrapped_layout.runs
    }

    /// The index corresponding to a given position in this layout for the given line height.
    ///
    /// See also [`Self::closest_index_for_position`].
    pub fn index_for_position(
        &self,
        position: Point<Pixels>,
        line_height: Pixels,
    ) -> Result<usize, usize> {
        self._index_for_position(position, line_height, false)
    }

    /// The closest index to a given position in this layout for the given line height.
    ///
    /// Closest means the character boundary closest to the given position.
    ///
    /// See also [`LineLayout::closest_index_for_x`].
    pub fn closest_index_for_position(
        &self,
        position: Point<Pixels>,
        line_height: Pixels,
    ) -> Result<usize, usize> {
        self._index_for_position(position, line_height, true)
    }

    fn _index_for_position(
        &self,
        position: Point<Pixels>,
        line_height: Pixels,
        closest: bool,
    ) -> Result<usize, usize> {
        let wrapped_line_ix = (position.y / line_height) as usize;

        let wrapped_line_start_index;
        let wrapped_line_start_x;
        if wrapped_line_ix > 0 {
            let Some(line_start_boundary) = self.wrap_boundaries.get(wrapped_line_ix - 1) else {
                return Err(0);
            };
            let run = &self.unwrapped_layout.runs[line_start_boundary.run_ix];
            let glyph = &run.glyphs[line_start_boundary.glyph_ix];
            wrapped_line_start_index = glyph.index;
            wrapped_line_start_x = glyph.position.x;
        } else {
            wrapped_line_start_index = 0;
            wrapped_line_start_x = Pixels::ZERO;
        };

        let wrapped_line_end_index;
        let wrapped_line_end_x;
        if wrapped_line_ix < self.wrap_boundaries.len() {
            let next_wrap_boundary_ix = wrapped_line_ix;
            let next_wrap_boundary = self.wrap_boundaries[next_wrap_boundary_ix];
            let run = &self.unwrapped_layout.runs[next_wrap_boundary.run_ix];
            let glyph = &run.glyphs[next_wrap_boundary.glyph_ix];
            wrapped_line_end_index = glyph.index;
            wrapped_line_end_x = glyph.position.x;
        } else {
            wrapped_line_end_index = self.unwrapped_layout.len;
            wrapped_line_end_x = self.unwrapped_layout.width;
        };

        let mut position_in_unwrapped_line = position;
        position_in_unwrapped_line.x += wrapped_line_start_x;
        if position_in_unwrapped_line.x < wrapped_line_start_x {
            Err(wrapped_line_start_index)
        } else if position_in_unwrapped_line.x >= wrapped_line_end_x {
            Err(wrapped_line_end_index)
        } else {
            if closest {
                Ok(self
                    .unwrapped_layout
                    .closest_index_for_x(position_in_unwrapped_line.x))
            } else {
                // The shaper can place a trailing zero-width wrap boundary glyph slightly past
                // the line's width, so the row can extend past where `index_for_x` has glyphs.
                self.unwrapped_layout
                    .index_for_x(position_in_unwrapped_line.x)
                    .ok_or(wrapped_line_end_index)
            }
        }
    }

    /// Returns the pixel position for the given byte index.
    pub fn position_for_index(&self, index: usize, line_height: Pixels) -> Option<Point<Pixels>> {
        let mut line_start_ix = 0;
        let line_end_indices = self
            .wrap_boundaries
            .iter()
            .map(|wrap_boundary| {
                let run = &self.unwrapped_layout.runs[wrap_boundary.run_ix];
                let glyph = &run.glyphs[wrap_boundary.glyph_ix];
                glyph.index
            })
            .chain([self.len()])
            .enumerate();
        for (ix, line_end_ix) in line_end_indices {
            let line_y = ix as f32 * line_height;
            if index < line_start_ix {
                break;
            } else if index > line_end_ix {
                line_start_ix = line_end_ix;
                continue;
            } else {
                let line_start_x = self.unwrapped_layout.x_for_index(line_start_ix);
                let x = self.unwrapped_layout.x_for_index(index) - line_start_x;
                return Some(point(x, line_y));
            }
        }

        None
    }
}

pub(crate) struct LineLayoutCache {
    state: RefCell<LineLayoutCacheState>,
    wrap_scratch: RefCell<WrapScratch>,
    platform_text_system: Arc<dyn PlatformTextSystem>,
    /// Advances when [`TextSystem::add_fonts`] successfully changes the font database.
    font_generation: Arc<AtomicUsize>,
    /// Records the generation represented by both frame caches.
    cached_font_generation: AtomicUsize,
}

#[derive(Default)]
pub(crate) struct WrapScratch {
    glyph_indices: FxHashSet<usize>,
    break_indices: Vec<usize>,
}

#[derive(Default)]
struct LineLayoutCacheState {
    previous_frame: FrameCache,
    current_frame: FrameCache,
}

#[derive(Default)]
struct FrameCache {
    lines: FxHashMap<Arc<CacheKey>, Arc<LineLayout>>,
    wrapped_lines: FxHashMap<Arc<CacheKey>, Arc<WrappedLineLayout>>,
    used_lines: Vec<Arc<CacheKey>>,
    used_wrapped_lines: Vec<Arc<CacheKey>>,
}

#[derive(Clone, Default)]
pub(crate) struct LineLayoutIndex {
    font_generation: usize,
    lines_index: usize,
    wrapped_lines_index: usize,
}

impl LineLayoutCache {
    pub fn new(
        platform_text_system: Arc<dyn PlatformTextSystem>,
        font_generation: Arc<AtomicUsize>,
    ) -> Self {
        let cached_font_generation = font_generation.load(Ordering::Acquire);
        Self {
            state: RefCell::default(),
            wrap_scratch: RefCell::default(),
            platform_text_system,
            font_generation,
            cached_font_generation: AtomicUsize::new(cached_font_generation),
        }
    }

    pub fn layout_index(&self) -> LineLayoutIndex {
        let font_generation = self.clear_if_font_generation_changed();
        let state = self.state.borrow();
        let frame = &state.current_frame;
        LineLayoutIndex {
            font_generation,
            lines_index: frame.used_lines.len(),
            wrapped_lines_index: frame.used_wrapped_lines.len(),
        }
    }

    pub fn reuse_layouts(&self, range: Range<LineLayoutIndex>) {
        let font_generation = self.clear_if_font_generation_changed();
        if range.start.font_generation != font_generation
            || range.end.font_generation != font_generation
        {
            return;
        }
        let mut state = self.state.borrow_mut();
        let LineLayoutCacheState {
            previous_frame,
            current_frame,
        } = &mut *state;

        for key in &previous_frame.used_lines[range.start.lines_index..range.end.lines_index] {
            if let Some((key, line)) = previous_frame.lines.remove_entry(key) {
                current_frame.lines.insert(key, line);
            }
            current_frame.used_lines.push(key.clone());
        }

        for key in &previous_frame.used_wrapped_lines
            [range.start.wrapped_lines_index..range.end.wrapped_lines_index]
        {
            if let Some((key, line)) = previous_frame.wrapped_lines.remove_entry(key) {
                current_frame.wrapped_lines.insert(key, line);
            }
            current_frame.used_wrapped_lines.push(key.clone());
        }
    }

    pub fn truncate_layouts(&self, index: LineLayoutIndex) {
        let font_generation = self.clear_if_font_generation_changed();
        if index.font_generation != font_generation {
            return;
        }
        let mut state = self.state.borrow_mut();
        let current_frame = &mut state.current_frame;
        current_frame.used_lines.truncate(index.lines_index);
        current_frame
            .used_wrapped_lines
            .truncate(index.wrapped_lines_index);
    }

    pub fn finish_frame(&self) {
        let _font_generation = self.clear_if_font_generation_changed();
        let LineLayoutCacheState {
            previous_frame,
            current_frame,
        } = &mut *self.state.borrow_mut();
        std::mem::swap(previous_frame, current_frame);
        current_frame.lines.clear();
        current_frame.wrapped_lines.clear();
        current_frame.used_lines.clear();
        current_frame.used_wrapped_lines.clear();
    }

    /// vue-native fork: the width of the hyphen a soft-hyphen break shows,
    /// in the first run's font. Zero when no break can show one.
    fn hyphen_width(
        &self,
        text: &str,
        font_size: Pixels,
        runs: &[FontRun],
        rules: TextWrap,
    ) -> Pixels {
        if rules.hyphens != Hyphens::Manual || !text.contains('\u{AD}') {
            return px(0.);
        }
        let Some(font_id) = runs.first().map(|run| run.font_id) else {
            return px(0.);
        };
        let Some(glyph) = self.platform_text_system.glyph_for_char(font_id, '-') else {
            return px(0.);
        };
        let Ok(advance) = self.platform_text_system.advance(font_id, glyph) else {
            return px(0.);
        };
        let units_per_em = self.platform_text_system.font_metrics(font_id).units_per_em as f32;
        px(advance.width / units_per_em * font_size.0)
    }

    pub fn layout_wrapped_line<Text>(
        &self,
        text: Text,
        font_size: Pixels,
        runs: &[FontRun],
        wrap_width: Option<Pixels>,
        max_lines: Option<usize>,
        rules: TextWrap,
    ) -> Arc<WrappedLineLayout>
    where
        Text: AsRef<str>,
        SharedString: From<Text>,
    {
        let _font_generation = self.clear_if_font_generation_changed();
        let key = &CacheKeyRef {
            text: text.as_ref(),
            font_size,
            runs,
            wrap_width,
            force_width: None,
            wrap: rules,
        } as &dyn AsCacheKeyRef;

        let state = self.state.borrow();
        if let Some(layout) = state.current_frame.wrapped_lines.get(key) {
            return layout.clone();
        }
        drop(state);

        let mut state = self.state.borrow_mut();
        let previous_frame_entry = state.previous_frame.wrapped_lines.remove_entry(key);
        if let Some((key, layout)) = previous_frame_entry {
            state
                .current_frame
                .wrapped_lines
                .insert(key.clone(), layout.clone());
            state.current_frame.used_wrapped_lines.push(key);
            layout
        } else {
            drop(state);
            let text = SharedString::from(text);
            let unwrapped_layout = self.layout_line::<&SharedString>(&text, font_size, runs, None);
            let wrapping = if let Some(wrap_width) = wrap_width {
                let hyphen_width = self.hyphen_width(&text, font_size, runs, rules);
                unwrapped_layout.wrap(
                    text.as_ref(),
                    wrap_width,
                    max_lines,
                    rules,
                    hyphen_width,
                    &mut self.wrap_scratch.borrow_mut(),
                )
            } else {
                Wrapping::default()
            };
            let layout = Arc::new(WrappedLineLayout {
                unwrapped_layout,
                wrap_boundaries: wrapping.boundaries,
                wrap_width,
                hyphen_breaks: wrapping.hyphens,
                overflow_width: wrapping.overflow,
            });
            let key = Arc::new(CacheKey {
                text,
                font_size,
                runs: SmallVec::from(runs),
                wrap_width,
                force_width: None,
                wrap: rules,
            });

            let mut state = self.state.borrow_mut();
            state
                .current_frame
                .wrapped_lines
                .insert(key.clone(), layout.clone());
            state.current_frame.used_wrapped_lines.push(key);

            layout
        }
    }

    pub fn layout_line<Text>(
        &self,
        text: Text,
        font_size: Pixels,
        runs: &[FontRun],
        force_width: Option<Pixels>,
    ) -> Arc<LineLayout>
    where
        Text: AsRef<str>,
        SharedString: From<Text>,
    {
        let _font_generation = self.clear_if_font_generation_changed();
        let key = &CacheKeyRef {
            text: text.as_ref(),
            font_size,
            runs,
            wrap_width: None,
            force_width,
            wrap: TextWrap::default(),
        } as &dyn AsCacheKeyRef;

        count_lookup();
        let state = self.state.borrow();
        if let Some(layout) = state.current_frame.lines.get(key) {
            return layout.clone();
        }
        drop(state);

        let mut state = self.state.borrow_mut();
        let LineLayoutCacheState {
            previous_frame,
            current_frame,
        } = &mut *state;
        if let Some((key, layout)) = previous_frame.lines.remove_entry(key) {
            current_frame.lines.insert(key.clone(), layout.clone());
            current_frame.used_lines.push(key);
            layout
        } else {
            let text = SharedString::from(text);
            let shaping = std::time::Instant::now();
            let mut layout = self
                .platform_text_system
                .layout_line(&text, font_size, runs);
            count_shape(shaping);

            // Spacing first: `force_width` assigns each cluster an ABSOLUTE
            // cell, so it has to see the positions it is going to overwrite,
            // not ones already widened by tracking.
            apply_spacing_to_layout(&mut layout, &text, runs);
            if let Some(force_width) = force_width {
                apply_force_width_to_layout(&mut layout, force_width);
            }

            let key = Arc::new(CacheKey {
                text,
                font_size,
                runs: SmallVec::from(runs),
                wrap_width: None,
                force_width,
                wrap: TextWrap::default(),
            });
            let layout = Arc::new(layout);
            current_frame.lines.insert(key.clone(), layout.clone());
            current_frame.used_lines.push(key);
            layout
        }
    }

    fn clear_if_font_generation_changed(&self) -> usize {
        let font_generation = self.font_generation.load(Ordering::Acquire);
        if self.cached_font_generation.load(Ordering::Acquire) == font_generation {
            return font_generation;
        }

        let mut state = self.state.borrow_mut();
        if self.cached_font_generation.load(Ordering::Acquire) == font_generation {
            return font_generation;
        }

        *state = LineLayoutCacheState::default();
        self.cached_font_generation
            .store(font_generation, Ordering::Release);
        font_generation
    }
}

/// CSS `letter-spacing` and `word-spacing`, applied to an already-shaped line.
///
/// A post-pass rather than a shaper feature, and deliberately so: every
/// platform text system gpui has (DirectWrite, CoreText, cosmic-text) hands
/// back absolute glyph POSITIONS, so widening the gaps between them is the same
/// arithmetic on all three and needs no per-backend work at all. Painting,
/// hit-testing, alignment, `split_at` and `compute_wrap_boundaries` read those
/// same positions and `LineLayout::width`, so they all follow for free.
///
/// The unit CSS spaces is the *typographic character unit*, not the glyph: a
/// combining mark or a ligature component must not be pushed away from what it
/// belongs to. `ShapedGlyph::index` is the cluster's start offset and every
/// glyph in a cluster shares it, so a change of `index` IS a cluster boundary.
///
/// Tracking is added AFTER each unit, including the last, which is what Chrome
/// does and what makes a tracked run measure `n * tracking` wider.
fn apply_spacing_to_layout(layout: &mut LineLayout, text: &str, runs: &[FontRun]) {
    if runs
        .iter()
        .all(|run| run.tracking == px(0.) && run.word_spacing == px(0.))
    {
        return;
    }

    // `runs` are byte spans over `text`, and a shaped run is not necessarily
    // the same span (the platform may split one on a font fallback), so the
    // spacing in force is looked up by the glyph's own byte offset.
    let spacing_at = |offset: usize| {
        let mut start = 0;
        for run in runs {
            if offset < start + run.len {
                return (run.tracking, run.word_spacing);
            }
            start += run.len;
        }
        runs.last()
            .map(|run| (run.tracking, run.word_spacing))
            .unwrap_or((px(0.), px(0.)))
    };

    let mut extra = px(0.);
    let mut previous_index: Option<usize> = None;
    for run in layout.runs.iter_mut() {
        for glyph in run.glyphs.iter_mut() {
            if previous_index != Some(glyph.index) {
                if let Some(previous) = previous_index {
                    let (tracking, word_spacing) = spacing_at(previous);
                    extra += tracking;
                    // CSS adds word spacing at each word-separator character,
                    // which for every script gpui shapes is U+0020.
                    if text[previous..].starts_with(' ') {
                        extra += word_spacing;
                    }
                }
                previous_index = Some(glyph.index);
            }
            glyph.position.x += extra;
        }
    }

    if let Some(previous) = previous_index {
        let (tracking, word_spacing) = spacing_at(previous);
        extra += tracking;
        if text[previous..].starts_with(' ') {
            extra += word_spacing;
        }
    }
    layout.width += extra;
}

// Combining marks (e.g. Thai vowel signs, Arabic diacritics) are shaped by
// HarfBuzz at the same x position as their base character. The force-width
// loop must not advance the cell counter for these zero-advance glyphs,
// otherwise they get displaced into the next cell. We detect them by checking
// whether shaped x has advanced by at least half a cell beyond the last base.
fn apply_force_width_to_layout(layout: &mut LineLayout, force_width: Pixels) {
    let mut glyph_pos: usize = 0;
    // NEG_INFINITY ensures the first glyph is always classified as a base.
    let mut last_base_shaped_x = px(f32::NEG_INFINITY);
    let mut last_base_actual_x = px(0.);

    for run in layout.runs.iter_mut() {
        for glyph in run.glyphs.iter_mut() {
            let shaped_x = glyph.position.x;

            if shaped_x > last_base_shaped_x + force_width * 0.5 {
                let forced_x = glyph_pos * force_width;
                if (shaped_x - forced_x).abs() > px(1.) {
                    glyph.position.x = forced_x;
                }
                last_base_shaped_x = shaped_x;
                last_base_actual_x = glyph.position.x;
                glyph_pos += 1;
            } else {
                glyph.position.x = last_base_actual_x + (shaped_x - last_base_shaped_x);
            }
        }
    }
}

/// A run of text with a single font.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[expect(missing_docs)]
pub struct FontRun {
    pub len: usize,
    pub font_id: FontId,
    /// Extra space after each typographic character unit — CSS
    /// `letter-spacing`. Part of the run, and therefore part of every layout
    /// CACHE KEY, because two runs that differ only in tracking are two
    /// different layouts.
    pub tracking: Pixels,
    /// Extra space after each word separator — CSS `word-spacing`.
    pub word_spacing: Pixels,
}

trait AsCacheKeyRef {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_>;
}

#[derive(Clone, Debug, Eq)]
struct CacheKey {
    text: SharedString,
    font_size: Pixels,
    runs: SmallVec<[FontRun; 1]>,
    wrap_width: Option<Pixels>,
    force_width: Option<Pixels>,
    /// vue-native fork: the same text wraps differently under other rules.
    wrap: TextWrap,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct CacheKeyRef<'a> {
    text: &'a str,
    font_size: Pixels,
    runs: &'a [FontRun],
    wrap_width: Option<Pixels>,
    force_width: Option<Pixels>,
    wrap: TextWrap,
}

impl PartialEq for dyn AsCacheKeyRef + '_ {
    fn eq(&self, other: &dyn AsCacheKeyRef) -> bool {
        self.as_cache_key_ref() == other.as_cache_key_ref()
    }
}

impl Eq for dyn AsCacheKeyRef + '_ {}

impl Hash for dyn AsCacheKeyRef + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_cache_key_ref().hash(state)
    }
}

impl AsCacheKeyRef for CacheKey {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_> {
        CacheKeyRef {
            text: &self.text,
            font_size: self.font_size,
            runs: self.runs.as_slice(),
            wrap_width: self.wrap_width,
            force_width: self.force_width,
            wrap: self.wrap,
        }
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_cache_key_ref().eq(&other.as_cache_key_ref())
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_cache_key_ref().hash(state);
    }
}

impl<'a> Borrow<dyn AsCacheKeyRef + 'a> for Arc<CacheKey> {
    fn borrow(&self) -> &(dyn AsCacheKeyRef + 'a) {
        self.as_ref() as &dyn AsCacheKeyRef
    }
}

impl AsCacheKeyRef for CacheKeyRef<'_> {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_> {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GlyphId, NoopTextSystem, OverflowWrap, TextRun, TextSystem, WindowTextSystem, font,
    };

    impl LineLayout {
        /// Upstream's wrapping entry point, as these tests call it: gpui's
        /// own rules ([`TextWrap::BREAK_WORD`]), no soft hyphens shown.
        fn compute_wrap_boundaries(
            &self,
            text: &str,
            wrap_width: Pixels,
            max_lines: Option<usize>,
            scratch: &mut WrapScratch,
        ) -> SmallVec<[WrapBoundary; 1]> {
            self.wrap(
                text,
                wrap_width,
                max_lines,
                TextWrap::BREAK_WORD,
                px(0.),
                scratch,
            )
            .boundaries
        }
    }

    #[test]
    fn shape_line_shares_its_text_with_the_layout_cache() {
        let text_system =
            WindowTextSystem::new(Arc::new(TextSystem::new(Arc::new(NoopTextSystem::new()))));
        // Long enough to be stored on the heap rather than inline.
        let text = SharedString::from("let value = compute_something(argument_one, argument_two);");
        let runs = [TextRun {
            len: text.len(),
            font: font(".ZedMono"),
            ..Default::default()
        }];

        text_system.shape_line(text.clone(), px(14.), &runs, None);
        let state = text_system.line_layout_cache.state.borrow();
        let (key, _) = state.current_frame.lines.iter().next().unwrap();
        assert_eq!(key.text.as_ptr(), text.as_ptr());
    }

    fn glyph_at(x: f32, index: usize) -> ShapedGlyph {
        ShapedGlyph {
            id: GlyphId(0),
            position: point(px(x), px(0.)),
            index,
            is_emoji: false,
        }
    }

    fn make_layout(glyphs: Vec<ShapedGlyph>) -> LineLayout {
        LineLayout {
            font_size: px(16.),
            width: px(100.),
            ascent: px(12.),
            descent: px(4.),
            runs: vec![ShapedRun {
                font_id: FontId(0),
                glyphs,
            }],
            len: 0,
        }
    }

    fn glyph_x_positions(layout: &LineLayout) -> Vec<f32> {
        layout.runs[0]
            .glyphs
            .iter()
            .map(|g| f32::from(g.position.x))
            .collect()
    }

    #[test]
    fn test_wrap_preserves_shaped_clusters() {
        let mut layout = make_layout(vec![
            glyph_at(0., 0),
            glyph_at(5., 0),
            glyph_at(10., 3),
            glyph_at(15., 3),
        ]);
        layout.width = px(20.);
        layout.len = "你好".len();
        assert_eq!(
            layout
                .compute_wrap_boundaries("你好", px(15.), None, &mut WrapScratch::default())
                .as_slice(),
            &[WrapBoundary {
                run_ix: 0,
                glyph_ix: 2
            }],
        );

        let mut layout = make_layout(vec![glyph_at(0., 0), glyph_at(10., 0)]);
        layout.width = px(20.);
        layout.len = 1;
        assert!(
            layout
                .compute_wrap_boundaries("a", px(10.), None, &mut WrapScratch::default())
                .is_empty()
        );
    }

    #[test]
    fn test_wrap_right_to_left_text() {
        for (text, indices, expected_boundaries) in [
            ("אבג דהו", vec![11, 9, 7, 6, 4, 2, 0], vec![4]),
            ("a אבג b", vec![0, 1, 6, 4, 2, 8, 9], vec![2, 6]),
        ] {
            let mut layout = make_layout(
                indices
                    .iter()
                    .enumerate()
                    .map(|(position, index)| glyph_at(position as f32 * 10., *index))
                    .collect(),
            );
            layout.width = px(indices.len() as f32 * 10.);
            layout.len = text.len();
            assert_eq!(
                layout.compute_wrap_boundaries(text, px(40.), None, &mut WrapScratch::default()),
                expected_boundaries
                    .into_iter()
                    .map(|glyph_ix| WrapBoundary {
                        run_ix: 0,
                        glyph_ix
                    })
                    .collect::<SmallVec<[WrapBoundary; 1]>>(),
                "{text:?}",
            );
        }
    }

    #[test]
    fn test_wrap_ignores_breaks_inside_shaped_clusters() {
        let text = "x a-bcde";
        let mut layout = make_layout(
            [0, 1, 2, 5, 6, 7]
                .into_iter()
                .map(|index| glyph_at(index as f32 * 10., index))
                .collect(),
        );
        layout.width = px(80.);
        layout.len = text.len();
        assert_eq!(
            layout
                .compute_wrap_boundaries(text, px(60.), None, &mut WrapScratch::default())
                .as_slice(),
            &[WrapBoundary {
                run_ix: 0,
                glyph_ix: 2
            }],
        );
    }

    #[test]
    fn test_force_width_latin_unchanged() {
        let cell_width = px(8.);
        let mut layout = make_layout(vec![glyph_at(0., 0), glyph_at(8., 1), glyph_at(16., 2)]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0., 8., 16.]);
    }

    #[test]
    fn test_force_width_combining_marks_not_advanced() {
        let cell_width = px(8.);
        // Simulates Thai "กี" — base consonant at x=0, combining vowel also at x=0
        let mut layout = make_layout(vec![
            glyph_at(0., 0), // ก (base)
            glyph_at(0., 3), // ี (combining mark, same x)
        ]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0., 0.]);
    }

    #[test]
    fn test_force_width_base_after_combining_mark() {
        let cell_width = px(8.);
        let mut layout = make_layout(vec![glyph_at(0., 0), glyph_at(0., 3), glyph_at(8., 6)]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0., 0., 8.]);
    }

    #[test]
    fn test_force_width_multiple_combining_marks() {
        let cell_width = px(8.);
        // Simulates "ก้" — base + vowel + tone mark (two combining marks stacked)
        let mut layout = make_layout(vec![
            glyph_at(0., 0), // ก (base)
            glyph_at(0., 3), // vowel (combining)
            glyph_at(0., 6), // tone mark (combining)
            glyph_at(8., 9), // next base
        ]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0., 0., 0., 8.]);
    }

    #[test]
    fn test_force_width_corrects_drifted_base_positions() {
        let cell_width = px(8.);
        // Font metrics don't perfectly match cell grid — glyphs drift >1px from cell boundary
        let mut layout = make_layout(vec![
            glyph_at(0.5, 0),  // within 1px tolerance, kept as-is
            glyph_at(10.2, 1), // >1px off from 8.0, corrected
            glyph_at(19.8, 2), // >1px off from 16.0, corrected
        ]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0.5, 8., 16.]);
    }

    #[test]
    fn test_force_width_combining_mark_after_within_tolerance_base() {
        let cell_width = px(8.);
        // Base glyph is within 1px of grid so it keeps its shaped position.
        // The combining mark must align to the base's actual position, not the grid slot.
        let mut layout = make_layout(vec![glyph_at(0.5, 0), glyph_at(0.5, 3)]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0.5, 0.5]);
    }

    /// One 10 px glyph per character of `text`; a soft hyphen and a zero
    /// width space take no room, as a shaper gives them.
    fn monospaced(text: &str) -> LineLayout {
        let mut x = 0.;
        let glyphs = text
            .char_indices()
            .map(|(i, ch)| {
                let glyph = glyph_at(x, i);
                if ch != '\u{AD}' && ch != '\u{200B}' {
                    x += 10.;
                }
                glyph
            })
            .collect();
        LineLayout {
            width: px(x),
            len: text.len(),
            ..make_layout(glyphs)
        }
    }

    fn boundary(glyph_ix: usize) -> WrapBoundary {
        WrapBoundary {
            run_ix: 0,
            glyph_ix,
        }
    }

    const HYPHEN: Pixels = px(10.);

    fn rules(overflow_wrap: OverflowWrap, word_break: WordBreak, hyphens: Hyphens) -> TextWrap {
        TextWrap {
            overflow_wrap,
            word_break,
            hyphens,
        }
    }

    const NORMAL: TextWrap = TextWrap {
        overflow_wrap: OverflowWrap::Normal,
        word_break: WordBreak::Normal,
        hyphens: Hyphens::Manual,
    };

    /// Where `text` breaks at `width`, and how wide its widest overflowing line is.
    fn wrap(text: &str, width: f32, rules: TextWrap) -> (Vec<usize>, Vec<usize>, Pixels) {
        let w = monospaced(text).wrap(
            text,
            px(width),
            None,
            rules,
            HYPHEN,
            &mut WrapScratch::default(),
        );
        (
            w.boundaries.iter().map(|b| b.glyph_ix).collect(),
            w.hyphens.to_vec(),
            w.overflow,
        )
    }

    fn min_content(text: &str, rules: TextWrap) -> Pixels {
        monospaced(text).min_content_width(text, rules, HYPHEN)
    }

    #[test]
    fn min_content_width_is_the_widest_word() {
        assert_eq!(min_content("ab cd efg", NORMAL), px(30.));
        // Spaces end a stretch and hang: they never count.
        assert_eq!(min_content("abc   de ", NORMAL), px(30.));
        // UAX #14: a break after a solidus, never before it ("a/" + "bcd").
        assert_eq!(min_content("a/bcd", NORMAL), px(30.));
        // One word: all of it.
        assert_eq!(min_content("abcdef", NORMAL), px(60.));
        assert_eq!(min_content("   ", NORMAL), px(0.));
    }

    #[test]
    fn wrapped_at_min_content_it_breaks_at_every_opportunity_and_in_no_word() {
        let text = "ab cd efg";
        let width = min_content(text, NORMAL);
        assert_eq!(wrap(text, width.0, NORMAL).0, vec![3, 6]);
    }

    #[test]
    fn a_space_at_the_edge_hangs_instead_of_starting_the_next_line() {
        // "ab" fills the 20 px line; the space after it crosses the edge. The
        // break goes before "cd", not before the space.
        assert_eq!(wrap("ab cd", 20., NORMAL).0, vec![3]);
    }

    /// CSS Text 3 §5.5. `normal`: a word wider than the line overflows it,
    /// and the line breaks at the next opportunity. `break-word` and
    /// `anywhere`: the word is broken where it overflows.
    #[test]
    fn overflow_wrap_decides_whether_a_long_word_breaks() {
        let text = "aaaaaaaa bb";
        assert_eq!(wrap(text, 30., NORMAL), (vec![9], vec![], px(80.)));
        let break_word = rules(OverflowWrap::BreakWord, WordBreak::Normal, Hyphens::Manual);
        assert_eq!(wrap(text, 30., break_word), (vec![3, 6, 9], vec![], px(0.)));
        let anywhere = rules(OverflowWrap::Anywhere, WordBreak::Normal, Hyphens::Manual);
        assert_eq!(wrap(text, 30., anywhere).0, vec![3, 6, 9]);
        // The legacy `word-break: break-word` is `overflow-wrap: anywhere`.
        let legacy = rules(OverflowWrap::Normal, WordBreak::BreakWord, Hyphens::Manual);
        assert_eq!(wrap(text, 30., legacy).0, vec![3, 6, 9]);
        // gpui's own callers keep breaking.
        assert_eq!(wrap(text, 30., TextWrap::BREAK_WORD).0, vec![3, 6, 9]);
        // A word that fits is never broken, under any of them.
        assert_eq!(wrap("ab cd", 30., anywhere).0, vec![3]);
    }

    /// `anywhere` lowers the min-content size to one character; `break-word`
    /// does not (CSS Text 3 §5.5).
    #[test]
    fn only_anywhere_counts_its_breaks_for_min_content() {
        let anywhere = rules(OverflowWrap::Anywhere, WordBreak::Normal, Hyphens::Manual);
        let break_word = rules(OverflowWrap::BreakWord, WordBreak::Normal, Hyphens::Manual);
        let legacy = rules(OverflowWrap::Normal, WordBreak::BreakWord, Hyphens::Manual);
        assert_eq!(min_content("abcdef gh", anywhere), px(10.));
        assert_eq!(min_content("abcdef gh", legacy), px(10.));
        assert_eq!(min_content("abcdef gh", break_word), px(60.));
        // Never inside a grapheme cluster: e + a combining acute is one.
        assert_eq!(min_content("ae\u{301}b", anywhere), px(20.));
    }

    /// `word-break: break-all` breaks between any two letters, so a line
    /// fills up instead of wrapping at its last space; `keep-all` keeps CJK
    /// letters together (CSS Text 3 §5.2).
    #[test]
    fn word_break_moves_the_opportunities() {
        let break_all = rules(OverflowWrap::Normal, WordBreak::BreakAll, Hyphens::Manual);
        assert_eq!(wrap("ab cdefgh", 50., break_all).0, vec![5]);
        let break_word = rules(OverflowWrap::BreakWord, WordBreak::Normal, Hyphens::Manual);
        assert_eq!(wrap("ab cdefgh", 50., break_word).0, vec![3, 8]);
        assert_eq!(min_content("abc", break_all), px(10.));
        // Not before punctuation that never starts a line.
        assert_eq!(min_content("a,b", break_all), px(20.));

        let keep_all = rules(OverflowWrap::Normal, WordBreak::KeepAll, Hyphens::Manual);
        assert_eq!(min_content("日本語", NORMAL), px(10.));
        assert_eq!(min_content("日本語", keep_all), px(30.));
        assert_eq!(min_content("日本 語", keep_all), px(20.));
    }

    /// A soft hyphen is a break opportunity under `hyphens: manual`, after
    /// it, with a hyphen shown that the line must fit; never under `none`
    /// (CSS Text 3 §6.1).
    #[test]
    fn a_soft_hyphen_breaks_with_a_hyphen() {
        let text = "ab\u{AD}cd";
        // "ab" and its hyphen fill the 30 px line.
        assert_eq!(wrap(text, 30., NORMAL), (vec![3], vec![0], px(0.)));
        assert_eq!(min_content(text, NORMAL), px(30.));
        // Without room for the hyphen, the break does not happen there.
        assert_eq!(wrap(text, 25., NORMAL).0, Vec::<usize>::new());
        let none = rules(OverflowWrap::Normal, WordBreak::Normal, Hyphens::None);
        assert_eq!(wrap(text, 30., none), (vec![], vec![], px(40.)));
        assert_eq!(min_content(text, none), px(40.));
    }

    #[test]
    fn a_zero_width_space_is_a_break_opportunity() {
        assert_eq!(wrap("aaa\u{200B}bbb", 30., NORMAL).0, vec![4]);
        assert_eq!(min_content("aaa\u{200B}bbb", NORMAL), px(30.));
    }
}
