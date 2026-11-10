//! The platform-neutral decisions behind the desktop picture a blurred window
//! shows in native fullscreen (`fullscreen_backdrop.rs` does the AppKit side).
//!
//! A fullscreen window has a Space of its own with nothing behind it, so the
//! behind-window blur of an `NSVisualEffectView` composites over black. In
//! fullscreen the window draws the desktop picture itself, under the blur
//! switched to within-window blending, so it looks as it did windowed.
//!
//! Nothing here touches AppKit, so it is unit-tested on every host.

/// What the blur view of a blurred window blurs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlurSource {
    /// Windowed: whatever is behind the window (behind-window blending).
    BehindWindow,
    /// Native fullscreen: the desktop picture, drawn inside the window under
    /// the blur (within-window blending).
    DesktopPicture,
    /// Native fullscreen with no readable desktop picture: the system window
    /// background, flat. The window reports its appearance as not drawn, so
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

/// The longest side, in pixels, to decode the picture at: as large as `fit`
/// draws it on a `screen`-sized window and never larger than the file, so a
/// 6K picture on a 1440p screen is decoded at 1440p. `0` means the sizes are
/// unusable and the picture should not be read.
pub(crate) fn decode_max_pixel_size(image: (f64, f64), screen: (f64, f64), fit: Fit) -> u32 {
    let (iw, ih) = image;
    let (sw, sh) = screen;
    let usable = |v: f64| v.is_finite() && v >= 1.0;
    if !(usable(iw) && usable(ih) && usable(sw) && usable(sh)) {
        return 0;
    }
    let scale = match fit {
        // Stretch scales each axis on its own; the decode keeps the aspect,
        // so it has to cover both, as Fill does.
        Fit::Fill | Fit::Stretch => (sw / iw).max(sh / ih),
        Fit::Fit => (sw / iw).min(sh / ih),
        Fit::Center => 1.0,
    }
    .min(1.0);
    let longest = (iw.max(ih) * scale).ceil();
    longest.clamp(1.0, u32::MAX as f64) as u32
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
    fn the_picture_is_decoded_no_larger_than_it_is_drawn() {
        let screen = (2880.0, 1800.0);
        // Same aspect: the screen's longest side.
        assert_eq!(
            decode_max_pixel_size((5760.0, 3600.0), screen, Fit::Fill),
            2880
        );
        // A wider picture filling a 16:10 screen is cropped at the sides, so
        // its height matches the screen's and its width runs past it.
        assert_eq!(
            decode_max_pixel_size((6300.0, 2700.0), screen, Fit::Fill),
            4200
        );
        assert_eq!(
            decode_max_pixel_size((6300.0, 2700.0), screen, Fit::Stretch),
            4200
        );
        // Fitted, the same picture is as wide as the screen.
        assert_eq!(
            decode_max_pixel_size((6300.0, 2700.0), screen, Fit::Fit),
            2880
        );
        // Never upscaled, and centred pictures keep their pixels.
        assert_eq!(
            decode_max_pixel_size((1024.0, 768.0), screen, Fit::Fill),
            1024
        );
        assert_eq!(
            decode_max_pixel_size((6000.0, 4000.0), screen, Fit::Center),
            6000
        );
        // Unusable sizes read nothing.
        assert_eq!(decode_max_pixel_size((0.0, 768.0), screen, Fit::Fill), 0);
        assert_eq!(
            decode_max_pixel_size((f64::NAN, 768.0), screen, Fit::Fill),
            0
        );
        assert_eq!(
            decode_max_pixel_size((1024.0, 768.0), (0.0, 0.0), Fit::Fill),
            0
        );
    }
}
