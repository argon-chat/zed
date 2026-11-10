//! The platform-neutral decisions behind the desktop picture a blurred window
//! shows in native fullscreen (`fullscreen_backdrop.rs` does the AppKit side).
//!
//! A fullscreen window has a Space of its own with nothing behind it, so the
//! behind-window blur of an `NSVisualEffectView` composites over black. In
//! fullscreen the window draws the desktop picture itself, under the blur
//! switched to within-window blending, so it looks as it did windowed.
//!
//! Here: which pictures may be read, how the desktop scales them, the size
//! to decode them at, the cache of decoded pictures shared by every window
//! (its key and bounds), and the colours that stand in for a picture.
//! Nothing here touches AppKit, so it is unit-tested on every host.

/// What the blur view of a blurred window blurs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlurSource {
    /// Windowed: whatever is behind the window (behind-window blending).
    BehindWindow,
    /// Native fullscreen: the desktop picture, drawn inside the window under
    /// the blur (within-window blending).
    DesktopPicture,
    /// Native fullscreen with no readable desktop picture: a flat neutral fill
    /// ([`neutral_fill`]). The window reports its appearance as not drawn, so
    /// the app can paint a background of its own.
    Flat,
}

impl BlurSource {
    /// Whether the window draws the blurred appearance it was given.
    pub(crate) fn draws_appearance(self) -> bool {
        self != BlurSource::Flat
    }
}

/// What a blurred window blurs, given whether it is in native fullscreen and
/// whether the desktop picture of its screen can be read.
pub(crate) fn blur_source(fullscreen: bool, picture_readable: bool) -> BlurSource {
    match (fullscreen, picture_readable) {
        (false, _) => BlurSource::BehindWindow,
        (true, true) => BlurSource::DesktopPicture,
        (true, false) => BlurSource::Flat,
    }
}

/// Folders under a home directory that macOS guards with a TCC prompt
/// ("… would like to access files in your Desktop folder"), compared without
/// case as APFS does by default. iCloud Drive lives in `Library/Mobile
/// Documents`, File Provider clouds in `Library/CloudStorage`, and other apps'
/// data in `Library/Containers` and `Library/Group Containers` (macOS 14+).
const PROTECTED_IN_HOME: &[&[&str]] = &[
    &["Desktop"],
    &["Documents"],
    &["Downloads"],
    &["Library", "Mobile Documents"],
    &["Library", "CloudStorage"],
    &["Library", "Containers"],
    &["Library", "Group Containers"],
];

/// Whether the desktop picture at `path` can be read without a privacy
/// prompt. A wallpaper is decoration: the window never asks for a folder
/// grant to draw one, and falls back to the flat background instead.
///
/// Allowed: the system pictures (`/System/Library/Desktop Pictures`,
/// `/Library/Desktop Pictures`), `~/Pictures` and the rest of `~/Library`.
/// Refused: `~/Desktop`, `~/Documents`, `~/Downloads`, iCloud Drive and other
/// cloud storage, other apps' containers, a Photos library, and anything on
/// a removable or network volume (`/Volumes`, `/Network`). `path` and `home`
/// (the user's home directory) are POSIX paths, whatever the host running the
/// tests; a path in any `/Users/<name>` gets the same rules as one in `home`.
pub(crate) fn may_read_without_prompt(path: &str, home: Option<&str>) -> bool {
    let Some(names) = absolute_names(path) else {
        return false;
    };
    // The data volume's firmlink spelling of the same folders.
    let names = match names.as_slice() {
        ["System", "Volumes", "Data", rest @ ..] => rest,
        all => all,
    };
    if matches!(names.first(), Some(&"Volumes") | Some(&"Network")) {
        return false;
    }
    if names
        .iter()
        .any(|n| n.to_ascii_lowercase().ends_with(".photoslibrary"))
    {
        return false;
    }
    let home = home
        .and_then(absolute_names)
        .filter(|home| !home.is_empty());
    let in_home = match (home, names) {
        (Some(home), names) if names.len() > home.len() && names[..home.len()] == home[..] => {
            &names[home.len()..]
        }
        (_, ["Users", _user, rest @ ..]) => rest,
        _ => return true,
    };
    !PROTECTED_IN_HOME.iter().any(|guarded| {
        in_home.len() > guarded.len()
            && guarded
                .iter()
                .zip(in_home)
                .all(|(g, n)| g.eq_ignore_ascii_case(n))
    })
}

/// The names in an absolute POSIX path, or `None` for a relative path or one
/// with `..`, which could climb out of an allowed folder into a guarded one.
fn absolute_names(path: &str) -> Option<Vec<&str>> {
    let rest = path.strip_prefix('/')?;
    rest.split('/')
        .filter(|n| !n.is_empty() && *n != ".")
        .map(|n| (n != "..").then_some(n))
        .collect()
}

/// How the desktop draws its picture, from System Settings › Wallpaper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fit {
    /// "Fill Screen": scaled to cover the screen, keeping its aspect, cropped.
    Fill,
    /// "Fit to Screen": scaled to fit inside the screen, the rest in the fill colour.
    Fit,
    /// "Stretch to Fill Screen": both axes scaled to the screen.
    Stretch,
    /// "Centre": unscaled, in the middle.
    Center,
}

impl Fit {
    /// The `CALayer.contentsGravity` that draws the picture this way.
    pub(crate) fn contents_gravity(self) -> &'static str {
        match self {
            Fit::Fill => "resizeAspectFill",
            Fit::Fit => "resizeAspect",
            Fit::Stretch => "resize",
            Fit::Center => "center",
        }
    }
}

// `NSImageScaling`, the value of `NSWorkspaceDesktopImageScalingKey`.
const NS_IMAGE_SCALE_PROPORTIONALLY_DOWN: i64 = 0;
const NS_IMAGE_SCALE_AXES_INDEPENDENTLY: i64 = 1;
const NS_IMAGE_SCALE_NONE: i64 = 2;
const NS_IMAGE_SCALE_PROPORTIONALLY_UP_OR_DOWN: i64 = 3;

/// The desktop's own [`Fit`], from `desktopImageOptionsForScreen:`: its
/// `NSWorkspaceDesktopImageScalingKey` (`scaling`) and
/// `NSWorkspaceDesktopImageAllowClippingKey` (`allow_clipping`), either of
/// which may be missing. `image` and `screen` are sizes in pixels.
///
/// Without options the desktop fills the screen, and clipping is allowed
/// unless the options say otherwise (AppKit's documented defaults).
pub(crate) fn desktop_fit(
    scaling: Option<i64>,
    allow_clipping: Option<bool>,
    image: (f64, f64),
    screen: (f64, f64),
) -> Fit {
    let clipping = allow_clipping.unwrap_or(true);
    match scaling {
        Some(NS_IMAGE_SCALE_AXES_INDEPENDENTLY) => Fit::Stretch,
        Some(NS_IMAGE_SCALE_NONE) => Fit::Center,
        // Only ever shrunk: a picture smaller than the screen stays as it is.
        Some(NS_IMAGE_SCALE_PROPORTIONALLY_DOWN) if image.0 <= screen.0 && image.1 <= screen.1 => {
            Fit::Center
        }
        Some(NS_IMAGE_SCALE_PROPORTIONALLY_DOWN | NS_IMAGE_SCALE_PROPORTIONALLY_UP_OR_DOWN)
        | None
        | Some(_) => {
            if clipping {
                Fit::Fill
            } else {
                Fit::Fit
            }
        }
    }
}

/// The picture is only ever seen through a heavy blur, so it is decoded at a
/// fraction of the screen's resolution and the layer scales it up (linearly
/// filtered): a quarter of the screen's longest pixel side...
pub(crate) const BLUR_DECODE_FRACTION: f64 = 0.25;
/// ...but never below this many pixels on the long side (unless the file is
/// smaller)...
pub(crate) const BLUR_DECODE_MIN: u32 = 256;
/// ...nor above this many, whatever the screen: at most ~4 MiB a picture.
pub(crate) const BLUR_DECODE_MAX: u32 = 1024;

/// The long side, in pixels, a picture filling a `screen` (in pixels) is
/// decoded at: [`BLUR_DECODE_FRACTION`] of the screen's longest side, within
/// [`BLUR_DECODE_MIN`]..=[`BLUR_DECODE_MAX`]. Part of the cache key.
pub(crate) fn blur_long_side(screen: (u32, u32)) -> u32 {
    let longest = screen.0.max(screen.1) as f64;
    ((longest * BLUR_DECODE_FRACTION).round() as u32).clamp(BLUR_DECODE_MIN, BLUR_DECODE_MAX)
}

/// The longest side, in pixels, to decode the picture at: the size `fit`
/// draws it at on a `screen`-sized window, times `target` (from
/// [`blur_long_side`]) over the screen's longest side, so every screen pixel
/// gets the same share of a decoded pixel whatever the picture's shape; never
/// above [`BLUR_DECODE_MAX`], never below [`BLUR_DECODE_MIN`], never larger
/// than the file. `0` means the sizes are unusable and the picture should not
/// be read.
pub(crate) fn decode_max_pixel_size(
    image: (f64, f64),
    screen: (f64, f64),
    fit: Fit,
    target: u32,
) -> u32 {
    let (iw, ih) = image;
    let (sw, sh) = screen;
    let usable = |v: f64| v.is_finite() && v >= 1.0;
    if !(usable(iw) && usable(ih) && usable(sw) && usable(sh)) || target == 0 {
        return 0;
    }
    let scale = match fit {
        // Stretch scales each axis on its own; the decode keeps the aspect,
        // so it has to cover both, as Fill does.
        Fit::Fill | Fit::Stretch => (sw / iw).max(sh / ih),
        Fit::Fit => (sw / iw).min(sh / ih),
        Fit::Center => 1.0,
    };
    let file = iw.max(ih);
    let drawn = file * scale;
    let decoded = (drawn * target as f64 / sw.max(sh))
        .ceil()
        .clamp(BLUR_DECODE_MIN as f64, BLUR_DECODE_MAX as f64)
        .min(file.ceil());
    decoded.max(1.0) as u32
}

/// The `CALayer.contentsScale` for a picture of `image` pixels decoded at
/// `decoded` pixels, on a screen of `backing` scale: its logical size stays
/// the file's size in screen pixels, as the desktop draws it. That is what
/// the `center` gravity draws at; the other gravities scale the contents to
/// the layer's bounds — the screen's logical size — and ignore it. So the
/// geometry is the desktop's however small the decode.
pub(crate) fn picture_contents_scale(image: (f64, f64), decoded: (f64, f64), backing: f64) -> f64 {
    let backing = if backing.is_finite() && backing > 0.0 {
        backing
    } else {
        1.0
    };
    let usable = |v: f64| v.is_finite() && v >= 1.0;
    if ![image.0, image.1, decoded.0, decoded.1]
        .into_iter()
        .all(usable)
    {
        return backing;
    }
    backing * decoded.0.max(decoded.1) / image.0.max(image.1)
}

/// An sRGB colour, each component in `0.0..=1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rgb {
    pub(crate) r: f64,
    pub(crate) g: f64,
    pub(crate) b: f64,
}

impl Rgb {
    pub(crate) const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Rec. 709 luma of the encoded components: how light the colour looks.
    pub(crate) fn luma(self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// The colour with `alpha`, as 8-bit components (for comparing colours
    /// read back from AppKit without float noise).
    pub(crate) fn to_rgba8(self, alpha: f64) -> [u8; 4] {
        let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        [q(self.r), q(self.g), q(self.b), q(alpha)]
    }
}

/// The average colour of a tiny downscale of the picture, as RGBA8 pixels
/// with premultiplied alpha (a `CGBitmapContext` with
/// `kCGImageAlphaPremultipliedLast`), weighted by coverage. `None` when the
/// pixels are empty or fully transparent.
///
/// It is what the blurred picture looks like from afar, so it stands in for
/// the picture wherever the picture is not (yet) drawn.
pub(crate) fn average_color(rgba: &[u8]) -> Option<Rgb> {
    let mut sum = [0u64; 3];
    let mut coverage = 0u64;
    for px in rgba.chunks_exact(4) {
        sum[0] += px[0] as u64;
        sum[1] += px[1] as u64;
        sum[2] += px[2] as u64;
        coverage += px[3] as u64;
    }
    if coverage == 0 {
        return None;
    }
    // Premultiplied: each stored component is `c * a`, so the sums over the
    // summed alpha are the coverage-weighted means of `c`.
    let mean = |s: u64| (s as f64 / coverage as f64).clamp(0.0, 1.0);
    Some(Rgb::new(mean(sum[0]), mean(sum[1]), mean(sum[2])))
}

/// `NSColor.windowBackgroundColor` in the light and dark Aqua appearances,
/// for when it cannot be resolved.
pub(crate) const LIGHT_WINDOW_BACKGROUND: Rgb = Rgb::new(0.925, 0.925, 0.925);
pub(crate) const DARK_WINDOW_BACKGROUND: Rgb = Rgb::new(0.196, 0.196, 0.196);

/// The lightest a fill without a picture gets. A window whose appearance is
/// light can belong to an app that paints itself dark (the app's theme is not
/// the window's `NSAppearance`), so the fill never goes to white: a light
/// window background is darkened to a neutral mid grey, which neither a light
/// nor a dark app reads as a flash. Dark backgrounds are already below it.
pub(crate) const NEUTRAL_MAX_LUMA: f64 = 0.45;

/// The fill under the blur when there is no picture to draw: the window's
/// background colour for its effective appearance (`window_background`, as
/// resolved by AppKit, or the Aqua default for `dark`), never lighter than
/// [`NEUTRAL_MAX_LUMA`].
pub(crate) fn neutral_fill(window_background: Option<Rgb>, dark: bool) -> Rgb {
    let base = window_background.unwrap_or(if dark {
        DARK_WINDOW_BACKGROUND
    } else {
        LIGHT_WINDOW_BACKGROUND
    });
    let luma = base.luma();
    if luma <= NEUTRAL_MAX_LUMA {
        return base;
    }
    let k = NEUTRAL_MAX_LUMA / luma;
    Rgb::new(base.r * k, base.g * k, base.b * k)
}

/// What identifies a decoded desktop picture: the screen (its
/// `CGDirectDisplayID`), the picture file, the desktop's options for it, and
/// the screen's size in pixels (a scale change re-decodes) and the long side
/// the blurred picture is decoded at for it ([`blur_long_side`]). Two windows
/// on one screen share it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PictureKey {
    pub(crate) display: u32,
    pub(crate) path: String,
    pub(crate) scaling: Option<i64>,
    pub(crate) allow_clipping: Option<bool>,
    pub(crate) fill: Option<[u8; 4]>,
    pub(crate) screen_px: (u32, u32),
    pub(crate) decode_long_side: u32,
}

/// What [`PictureCache::request`] found.
#[derive(Debug, PartialEq)]
pub(crate) enum Lookup<V> {
    /// Decoded before: here it is.
    Hit(V),
    /// Being decoded already; the waiter, if any, is told when it lands.
    Joined,
    /// Not decoded nor being decoded: the caller starts the decode and calls
    /// [`PictureCache::finish`] with its result.
    Started,
}

struct Entry<V> {
    key: PictureKey,
    value: V,
    bytes: usize,
}

/// Decoded desktop pictures, shared by every window, newest first, and the
/// decodes in flight with whoever waits for them.
///
/// Bounded: one entry per screen (a newer picture for a screen replaces the
/// older one once decoded), at most `max_entries` screens, and at most
/// `max_bytes` of pixels — the newest entry is always kept, so the screen in
/// use never loses its picture to the budget. `V` is the decoded picture,
/// `W` a waiter, neither of which this type looks into.
pub(crate) struct PictureCache<V, W> {
    entries: Vec<Entry<V>>,
    pending: Vec<(PictureKey, Vec<W>)>,
    max_entries: usize,
    max_bytes: usize,
}

impl<V: Clone, W> PictureCache<V, W> {
    pub(crate) const fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            pending: Vec::new(),
            max_entries,
            max_bytes,
        }
    }

    /// The picture decoded for `key`, now the most recently used.
    pub(crate) fn get(&mut self, key: &PictureKey) -> Option<V> {
        let index = self.entries.iter().position(|e| &e.key == key)?;
        let entry = self.entries.remove(index);
        let value = entry.value.clone();
        self.entries.insert(0, entry);
        Some(value)
    }

    /// The most recent `usable` picture decoded for `display`, whatever its
    /// file or options: a stale picture to show while the current one decodes.
    pub(crate) fn latest_for_display(
        &self,
        display: u32,
        usable: impl Fn(&V) -> bool,
    ) -> Option<V> {
        self.entries
            .iter()
            .find(|e| e.key.display == display && usable(&e.value))
            .map(|e| e.value.clone())
    }

    /// Looks `key` up: decoded ([`Lookup::Hit`], `waiter` dropped), being
    /// decoded (`waiter` joins), or neither (`waiter` waits for the decode
    /// the caller now starts).
    pub(crate) fn request(&mut self, key: &PictureKey, waiter: Option<W>) -> Lookup<V> {
        if let Some(value) = self.get(key) {
            return Lookup::Hit(value);
        }
        if let Some((_, waiters)) = self.pending.iter_mut().find(|(k, _)| k == key) {
            waiters.extend(waiter);
            return Lookup::Joined;
        }
        self.pending
            .push((key.clone(), waiter.into_iter().collect()));
        Lookup::Started
    }

    /// Records the decode of `key` (`bytes` of pixels) and returns its
    /// waiters, to be told after the cache is released.
    pub(crate) fn finish(&mut self, key: PictureKey, value: V, bytes: usize) -> Vec<W> {
        let waiters = match self.pending.iter().position(|(k, _)| *k == key) {
            Some(index) => self.pending.remove(index).1,
            None => Vec::new(),
        };
        self.insert(key, value, bytes);
        waiters
    }

    /// Adds a decoded picture as the most recent, dropping the screen's older
    /// one and whatever falls outside the bounds.
    pub(crate) fn insert(&mut self, key: PictureKey, value: V, bytes: usize) {
        self.entries.retain(|e| e.key.display != key.display);
        self.entries.insert(0, Entry { key, value, bytes });
        self.entries.truncate(self.max_entries.max(1));
        while self.entries.len() > 1 && self.total_bytes() > self.max_bytes {
            self.entries.pop();
        }
    }

    /// Drops every decoded picture (memory pressure). Decodes in flight and
    /// their waiters stay; pictures in use stay alive with their users.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn total_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blurred_window_blurs_the_picture_only_in_fullscreen() {
        assert_eq!(blur_source(false, true), BlurSource::BehindWindow);
        assert_eq!(blur_source(false, false), BlurSource::BehindWindow);
        assert_eq!(blur_source(true, true), BlurSource::DesktopPicture);
        assert_eq!(blur_source(true, false), BlurSource::Flat);
        assert!(BlurSource::BehindWindow.draws_appearance());
        assert!(BlurSource::DesktopPicture.draws_appearance());
        assert!(!BlurSource::Flat.draws_appearance());
    }

    fn readable(path: &str) -> bool {
        may_read_without_prompt(path, Some("/Users/ann/"))
    }

    #[test]
    fn system_pictures_and_the_users_library_are_read() {
        assert!(readable("/System/Library/Desktop Pictures/Sequoia.heic"));
        assert!(readable("/Library/Desktop Pictures/Company.jpg"));
        assert!(readable(
            "/System/Volumes/Data/Library/Desktop Pictures/x.png"
        ));
        assert!(readable("/Users/ann/Pictures/beach.jpg"));
        assert!(readable(
            "/Users/ann/Library/Application Support/com.apple.wallpaper/x.heic"
        ));
        assert!(readable("/Users/Shared/wall.png"));
    }

    #[test]
    fn guarded_folders_are_not_read() {
        for path in [
            "/Users/ann/Desktop/wall.png",
            "/Users/ann/desktop/wall.png",
            "/Users/ann/Documents/a/b.jpg",
            "/Users/ann/Downloads/x.heic",
            "/Users/ann/Library/Mobile Documents/com~apple~CloudDocs/w.jpg",
            "/Users/ann/Library/CloudStorage/Dropbox/w.jpg",
            "/Users/ann/Library/Containers/com.apple.Photos/Data/w.jpg",
            "/Users/ann/Library/Group Containers/group.x/w.jpg",
            "/Users/ann/Pictures/Photos Library.photoslibrary/originals/1.heic",
            "/Volumes/USB/wall.jpg",
            "/Volumes/Macintosh HD/Users/ann/Pictures/x.jpg",
            "/Network/Servers/nas/wall.jpg",
            // Another user's guarded folder, and the firmlinked spelling.
            "/Users/bob/Desktop/wall.png",
            "/System/Volumes/Data/Users/ann/Documents/w.png",
            // Climbing out of an allowed folder.
            "/Users/ann/Pictures/../Desktop/w.png",
            "relative/wall.png",
        ] {
            assert!(!readable(path), "{path} must not be read");
        }
    }

    #[test]
    fn a_guarded_folder_itself_or_a_lookalike_name_is_not_confused() {
        // A file directly in the home, or a folder that only starts like one.
        assert!(readable("/Users/ann/wall.png"));
        assert!(readable("/Users/ann/Desktop Pictures/wall.png"));
        assert!(readable("/Users/ann/Library/ContainersBackup/w.png"));
    }

    #[test]
    fn without_a_home_the_users_folders_are_still_guarded() {
        assert!(!may_read_without_prompt("/Users/ann/Desktop/w.png", None));
        assert!(may_read_without_prompt(
            "/System/Library/Desktop Pictures/a.heic",
            None
        ));
        // A home outside /Users (a network or custom home) is guarded by `home`.
        assert!(!may_read_without_prompt(
            "/home/ann/Desktop/w.png",
            Some("/home/ann")
        ));
        assert!(may_read_without_prompt(
            "/home/ann/Pictures/w.png",
            Some("/home/ann")
        ));
    }

    #[test]
    fn the_desktop_fit_follows_its_options() {
        let big = (6000.0, 4000.0);
        let screen = (2880.0, 1800.0);
        assert_eq!(desktop_fit(None, None, big, screen), Fit::Fill);
        assert_eq!(desktop_fit(Some(3), Some(true), big, screen), Fit::Fill);
        assert_eq!(desktop_fit(Some(3), None, big, screen), Fit::Fill);
        assert_eq!(desktop_fit(Some(3), Some(false), big, screen), Fit::Fit);
        assert_eq!(desktop_fit(Some(1), None, big, screen), Fit::Stretch);
        assert_eq!(desktop_fit(Some(2), None, big, screen), Fit::Center);
        assert_eq!(desktop_fit(Some(0), Some(false), big, screen), Fit::Fit);
        assert_eq!(
            desktop_fit(Some(0), None, (800.0, 600.0), screen),
            Fit::Center
        );
        assert_eq!(desktop_fit(Some(42), Some(false), big, screen), Fit::Fit);
        assert_eq!(Fit::Fill.contents_gravity(), "resizeAspectFill");
        assert_eq!(Fit::Center.contents_gravity(), "center");
    }

    #[test]
    fn the_blurred_picture_is_decoded_at_a_quarter_of_the_screen() {
        assert_eq!(blur_long_side((2880, 1800)), 720);
        assert_eq!(blur_long_side((1800, 2880)), 720);
        assert_eq!(blur_long_side((1440, 900)), 360);
        assert_eq!(blur_long_side((3024, 1964)), 756);
        // A 5K or 6K screen is capped, a small one floored.
        assert_eq!(blur_long_side((5120, 2880)), BLUR_DECODE_MAX);
        assert_eq!(blur_long_side((6016, 3384)), BLUR_DECODE_MAX);
        assert_eq!(blur_long_side((800, 600)), BLUR_DECODE_MIN);
        assert_eq!(blur_long_side((0, 0)), BLUR_DECODE_MIN);
    }

    #[test]
    fn the_picture_is_decoded_small_in_proportion_to_how_it_is_drawn() {
        let screen = (2880.0, 1800.0);
        let decode = |image, fit| decode_max_pixel_size(image, screen, fit, 720);
        // Same aspect: a quarter of the screen's longest side.
        assert_eq!(decode((5760.0, 3600.0), Fit::Fill), 720);
        // A wider picture filling a 16:10 screen is cropped at the sides: its
        // height matches the screen's, its width runs past it (4200 px drawn,
        // 1050 at a quarter), capped.
        assert_eq!(decode((6300.0, 2700.0), Fit::Fill), BLUR_DECODE_MAX);
        assert_eq!(decode((6300.0, 2700.0), Fit::Stretch), BLUR_DECODE_MAX);
        // Fitted, the same picture is as wide as the screen.
        assert_eq!(decode((6300.0, 2700.0), Fit::Fit), 720);
        // A picture smaller than the screen but larger than the decode.
        assert_eq!(decode((1024.0, 768.0), Fit::Fill), 720);
        // Centred pictures keep their size on screen, so a large one is
        // decoded as large as the cap allows, a small one at the floor, and a
        // tiny one at its own size: never upscaled.
        assert_eq!(decode((6000.0, 4000.0), Fit::Center), BLUR_DECODE_MAX);
        assert_eq!(decode((2000.0, 1000.0), Fit::Center), 500);
        assert_eq!(decode((400.0, 300.0), Fit::Center), BLUR_DECODE_MIN);
        assert_eq!(decode((200.0, 150.0), Fit::Center), 200);
        assert_eq!(decode((200.0, 150.0), Fit::Fill), 200);
        // The target follows the screen.
        let five_k = (5120.0, 2880.0);
        let target = blur_long_side((5120, 2880));
        assert_eq!(
            decode_max_pixel_size((6016.0, 3384.0), five_k, Fit::Fill, target),
            BLUR_DECODE_MAX
        );
        let small = (1280.0, 800.0);
        let target = blur_long_side((1280, 800));
        assert_eq!(target, 320);
        assert_eq!(
            decode_max_pixel_size((5760.0, 3600.0), small, Fit::Fill, target),
            320
        );
        // Never above the cap, whatever the target.
        assert_eq!(
            decode_max_pixel_size((8000.0, 5000.0), screen, Fit::Fill, 5000),
            BLUR_DECODE_MAX
        );
        // Unusable sizes read nothing.
        assert_eq!(decode((0.0, 768.0), Fit::Fill), 0);
        assert_eq!(decode((f64::NAN, 768.0), Fit::Fill), 0);
        assert_eq!(
            decode_max_pixel_size((1024.0, 768.0), (0.0, 0.0), Fit::Fill, 720),
            0
        );
        assert_eq!(
            decode_max_pixel_size((1024.0, 768.0), screen, Fit::Fill, 0),
            0
        );
    }

    #[test]
    fn a_small_decode_keeps_the_desktops_geometry() {
        // A 6000 px picture centred on a 2x screen covers 3000 points; decoded
        // at 1024 px, the layer's contents scale keeps it at 3000 points.
        let scale = picture_contents_scale((6000.0, 4000.0), (1024.0, 683.0), 2.0);
        assert!((1024.0 / scale - 3000.0).abs() < 1e-9);
        // A picture decoded at its own size keeps the screen's scale.
        assert_eq!(
            picture_contents_scale((200.0, 150.0), (200.0, 150.0), 2.0),
            2.0
        );
        // Portrait pictures: the long side decides.
        let scale = picture_contents_scale((2000.0, 4000.0), (256.0, 512.0), 1.0);
        assert!((512.0 / scale - 4000.0).abs() < 1e-9);
        // Unusable sizes fall back to the screen's scale.
        assert_eq!(picture_contents_scale((0.0, 0.0), (10.0, 10.0), 2.0), 2.0);
        assert_eq!(
            picture_contents_scale((10.0, 10.0), (f64::NAN, 1.0), 2.0),
            2.0
        );
        assert_eq!(picture_contents_scale((10.0, 10.0), (5.0, 5.0), 0.0), 0.5);
    }

    fn close(a: Rgb, b: Rgb) -> bool {
        (a.r - b.r).abs() < 1e-3 && (a.g - b.g).abs() < 1e-3 && (a.b - b.b).abs() < 1e-3
    }

    #[test]
    fn the_average_colour_weighs_pixels_by_their_coverage() {
        // A flat colour averages to itself.
        let flat = [51u8, 102, 204, 255].repeat(16);
        assert!(close(
            average_color(&flat).unwrap(),
            Rgb::new(0.2, 0.4, 0.8)
        ));
        // Half black, half white.
        let mut halves = [0u8, 0, 0, 255].repeat(8);
        halves.extend([255u8, 255, 255, 255].repeat(8));
        assert!(close(
            average_color(&halves).unwrap(),
            Rgb::new(0.5, 0.5, 0.5)
        ));
        // Premultiplied: a half-transparent red pixel is stored as (128, 0, 0, 128)
        // and counts half as much as an opaque blue one.
        let mixed = [128u8, 0, 0, 128, 0, 0, 255, 255];
        let avg = average_color(&mixed).unwrap();
        assert!(close(avg, Rgb::new(128.0 / 383.0, 0.0, 255.0 / 383.0)));
        // Transparent or empty: nothing to average.
        assert_eq!(average_color(&[0u8, 0, 0, 0].repeat(4)), None);
        assert_eq!(average_color(&[]), None);
        // A trailing partial pixel is ignored.
        assert!(close(
            average_color(&[255u8, 0, 0, 255, 9]).unwrap(),
            Rgb::new(1.0, 0.0, 0.0)
        ));
    }

    #[test]
    fn the_fill_without_a_picture_is_never_white() {
        // Light appearance: the near-white window background becomes a mid grey.
        let light = neutral_fill(Some(Rgb::new(0.925, 0.925, 0.925)), false);
        assert!((light.luma() - NEUTRAL_MAX_LUMA).abs() < 1e-9);
        assert!(light.r == light.g && light.g == light.b);
        let fallback = neutral_fill(None, false);
        assert!(fallback.luma() <= NEUTRAL_MAX_LUMA + 1e-9);
        // White itself, and a tinted light colour keeps its hue.
        assert!(
            neutral_fill(Some(Rgb::new(1.0, 1.0, 1.0)), false).luma() <= NEUTRAL_MAX_LUMA + 1e-9
        );
        let tinted = neutral_fill(Some(Rgb::new(0.9, 0.95, 1.0)), false);
        assert!(tinted.r < tinted.g && tinted.g < tinted.b);
        // Dark appearance: the dark window background as it is.
        assert_eq!(neutral_fill(None, true), DARK_WINDOW_BACKGROUND);
        let dark = Rgb::new(0.12, 0.12, 0.13);
        assert_eq!(neutral_fill(Some(dark), true), dark);
        assert_eq!(Rgb::new(1.0, 0.5, 0.0).to_rgba8(1.0), [255, 128, 0, 255]);
        assert_eq!(Rgb::new(2.0, -1.0, 0.0).to_rgba8(0.0), [255, 0, 0, 0]);
    }

    fn key(display: u32, path: &str) -> PictureKey {
        PictureKey {
            display,
            path: path.into(),
            scaling: Some(3),
            allow_clipping: Some(true),
            fill: None,
            screen_px: (2880, 1800),
            decode_long_side: 720,
        }
    }

    #[test]
    fn a_picture_is_decoded_once_and_shared() {
        let mut cache: PictureCache<&str, u32> = PictureCache::new(3, 1000);
        let a = key(1, "/a.heic");
        // The first window starts the decode; the second joins it.
        assert_eq!(cache.request(&a, Some(1)), Lookup::Started);
        assert_eq!(cache.request(&a, Some(2)), Lookup::Joined);
        assert_eq!(cache.request(&a, None), Lookup::Joined);
        assert_eq!(cache.finish(a.clone(), "A", 10), vec![1, 2]);
        // Decoded: every later request is answered at once.
        assert_eq!(cache.request(&a, Some(3)), Lookup::Hit("A"));
        assert_eq!(cache.get(&a), Some("A"));
        // A finished decode nobody waits for (a prefetch) still caches.
        let b = key(2, "/b.jpg");
        assert_eq!(cache.request(&b, None), Lookup::Started);
        assert_eq!(cache.finish(b.clone(), "B", 10), Vec::<u32>::new());
        assert_eq!(cache.get(&b), Some("B"));
    }

    #[test]
    fn any_change_of_file_options_or_size_is_another_picture() {
        let mut cache: PictureCache<&str, ()> = PictureCache::new(3, 1000);
        let a = key(1, "/a.heic");
        cache.insert(a.clone(), "A", 10);
        let changes = [
            PictureKey {
                path: "/b.heic".into(),
                ..a.clone()
            },
            PictureKey {
                scaling: Some(0),
                ..a.clone()
            },
            PictureKey {
                allow_clipping: Some(false),
                ..a.clone()
            },
            PictureKey {
                fill: Some([0, 0, 0, 255]),
                ..a.clone()
            },
            PictureKey {
                screen_px: (5120, 2880),
                ..a.clone()
            },
            PictureKey {
                decode_long_side: 256,
                ..a.clone()
            },
            PictureKey {
                display: 2,
                ..a.clone()
            },
        ];
        for changed in &changes {
            assert_eq!(cache.get(changed), None, "{changed:?}");
        }
        // Stale for its screen, it is still what that screen showed last.
        assert_eq!(cache.latest_for_display(1, |_| true), Some("A"));
        assert_eq!(cache.latest_for_display(1, |v| *v != "A"), None);
        assert_eq!(cache.latest_for_display(2, |_| true), None);
    }

    #[test]
    fn the_cache_keeps_one_picture_per_screen_for_the_last_screens() {
        let mut cache: PictureCache<&str, ()> = PictureCache::new(3, 1000);
        cache.insert(key(1, "/old.heic"), "old", 10);
        // A new picture for screen 1 replaces the old one.
        cache.insert(key(1, "/new.heic"), "new", 10);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key(1, "/old.heic")), None);
        cache.insert(key(2, "/b"), "b", 10);
        cache.insert(key(3, "/c"), "c", 10);
        // Screen 1 used last, so screen 2 is the least recently used.
        assert_eq!(cache.get(&key(1, "/new.heic")), Some("new"));
        cache.insert(key(4, "/d"), "d", 10);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&key(2, "/b")), None);
        assert_eq!(cache.get(&key(3, "/c")), Some("c"));
        assert_eq!(cache.get(&key(1, "/new.heic")), Some("new"));
        // Memory pressure drops everything.
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn the_byte_budget_drops_old_pictures_but_never_the_newest() {
        let mut cache: PictureCache<&str, ()> = PictureCache::new(3, 100);
        cache.insert(key(1, "/a"), "a", 40);
        cache.insert(key(2, "/b"), "b", 40);
        assert_eq!(cache.len(), 2);
        cache.insert(key(3, "/c"), "c", 40);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&key(1, "/a")), None);
        assert_eq!(cache.total_bytes(), 80);
        // One picture larger than the whole budget is still kept, alone.
        cache.insert(key(4, "/huge"), "huge", 500);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key(4, "/huge")), Some("huge"));
        // ...until something newer arrives.
        cache.insert(key(5, "/e"), "e", 40);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key(5, "/e")), Some("e"));
        // A failed decode (no pixels) costs nothing.
        cache.insert(key(6, "/bad"), "bad", 0);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_bytes(), 40);
    }

    #[test]
    fn a_cleared_cache_still_delivers_the_decodes_in_flight() {
        let mut cache: PictureCache<&str, u32> = PictureCache::new(3, 1000);
        let a = key(1, "/a");
        assert_eq!(cache.request(&a, Some(7)), Lookup::Started);
        cache.clear();
        assert_eq!(cache.request(&a, Some(8)), Lookup::Joined);
        assert_eq!(cache.finish(a.clone(), "A", 1), vec![7, 8]);
        // A result nobody asked for (raced with a clear) is simply cached.
        assert_eq!(cache.finish(key(2, "/b"), "B", 1), Vec::<u32>::new());
        assert_eq!(cache.len(), 2);
    }
}
