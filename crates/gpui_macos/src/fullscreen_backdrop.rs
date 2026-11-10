//! A blurred window in native fullscreen: the desktop picture behind the blur.
//!
//! `WindowBackgroundAppearance::Blurred` is an `NSVisualEffectView` blending
//! with what is behind the window. A fullscreen window has a Space of its own
//! with nothing behind it, so that blur turns black. While a blurred window
//! is in native fullscreen it draws its screen's desktop picture itself — a
//! layer-hosting view under the blur view, scaled as the desktop scales it —
//! and the blur view blends within the window instead, so the glass looks as
//! it did windowed. Leaving fullscreen takes the picture away and restores
//! behind-window blending.
//!
//! # No flash on the way in or out
//!
//! AppKit's fullscreen transition runs right after `windowWillEnterFullScreen:`
//! returns: it moves the window into a new Space at the screen's size and
//! animates a snapshot of the windowed window into it, cross-fading to the
//! real window, then posts `windowDidEnterFullScreen:`. Whatever the real
//! window shows from `willEnter` on is on screen for the whole animation, so
//! the picture has to be there *at* `willEnter`, not a decode later:
//!
//! - **Decoded ahead.** A blurred window decodes its screen's picture off the
//!   main thread while it is still windowed — when it becomes blurred, when it
//!   moves to another screen, when its screen's scale changes — into a cache
//!   shared by every window ([`PictureCache`], one entry per screen, the last
//!   three screens, an 8 MiB budget, emptied on memory pressure). The picture
//!   is only ever seen through the heavy blur, so it is decoded at about a
//!   quarter of the screen's resolution (256..=1024 px on its long side,
//!   [`decode_max_pixel_size`]) — about 1.3 MiB for a 2880×1800 screen — and
//!   the layer scales it up with linear filtering, keeping the desktop's
//!   geometry ([`picture_contents_scale`]). The decode also averages a tiny
//!   downscale of the picture into one colour.
//! - **Installed at once.** At `willEnter` the picture comes from the cache:
//!   its `CGImage` goes into the picture layer, the blur switches to
//!   within-window blending and the window background takes the picture's
//!   average colour, all in one `CATransaction` with actions disabled — the
//!   first frame of the transition already shows the blurred picture, and no
//!   layer fades in. The transaction is flushed before `willEnter` returns, so
//!   it is committed before AppKit snapshots the window and composites the
//!   animation's first frame. Nothing is decoded on the main thread.
//! - **Stale, then fresh.** The cache is keyed by screen, file, desktop options,
//!   screen pixel size and decode size, and checked at `willEnter`. A changed wallpaper (or
//!   scale) shows the screen's previous picture during the animation and swaps
//!   to the new one, again in one transaction, when its decode lands.
//! - **The blur keeps its backing.** With nothing cached for the screen, the
//!   blur keeps blending behind the window — what the windowed window showed —
//!   until the decode lands or the transition ends (`windowDidEnterFullScreen:`),
//!   whichever comes first. Only then does it blend within the window, over the
//!   picture or, at `didEnter` without one, over a neutral fill. It never drops
//!   to a bare background mid-animation.
//! - **Never white.** Under the picture, and wherever there is none yet, the
//!   window background is the picture's average colour; without a picture it
//!   is the window background colour of the window's effective appearance,
//!   darkened to a mid grey when that is light ([`neutral_fill`]) — a dark app
//!   in a light window must not flash white.
//! - **Out.** The exit animation lands on the picture the window showed; at
//!   `windowDidExitFullScreen:` the picture goes, behind-window blending and
//!   the windowed background return, in one transaction.
//!
//! A picture that cannot be shown — none, one behind a privacy prompt
//! ([`may_read_without_prompt`]), a dynamic or aerial wallpaper ImageIO cannot
//! read, a failed decode — leaves the neutral fill, and the window reports its
//! appearance as not drawn (`PlatformWindow::is_background_appearance_drawn`),
//! so the app can paint a background of its own.
//!
//! Layer order inside the content view, bottom to top: the picture, the blur
//! view, gpui's view and its `CAMetalLayer`.

use crate::{
    NSStringExt,
    desktop_picture::{
        BlurSource, Fit, Lookup, PictureCache, PictureKey, Rgb, average_color, blur_long_side,
        blur_source, decode_max_pixel_size, desktop_fit, may_read_without_prompt, neutral_fill,
        picture_contents_scale,
    },
    ns_string,
    window_appearance::{NSAppearanceNameAqua, NSAppearanceNameDarkAqua},
};
use block2::RcBlock;
use cocoa::{
    appkit::{
        NSColor, NSScreen, NSView, NSViewHeightSizable, NSViewWidthSizable,
        NSVisualEffectBlendingMode, NSVisualEffectView, NSWindow, NSWindowOrderingMode,
    },
    base::{id, nil},
    foundation::{NSArray, NSInteger, NSRect},
};
use core_foundation::{
    base::{CFType, TCFType},
    boolean::CFBoolean,
    dictionary::CFDictionary,
    number::CFNumber,
    string::CFString,
};
use core_foundation_sys::{
    base::{CFGetTypeID, CFRelease, CFRetain},
    dictionary::{CFDictionaryGetValue, CFDictionaryRef},
    number::{CFNumberGetTypeID, CFNumberGetValue, CFNumberRef, kCFNumberFloat64Type},
    string::CFStringRef,
    url::CFURLRef,
};
use core_graphics::geometry::{CGPoint, CGRect, CGSize};
use dispatch2::{
    _dispatch_source_type_memorypressure, DispatchObject, DispatchQueue, DispatchRetained,
    DispatchSource,
};
use gpui::{BackgroundExecutor, ForegroundExecutor};
use objc::{
    class, msg_send,
    runtime::{BOOL, YES},
    sel, sel_impl,
};
use std::{
    cell::{Cell, OnceCell, RefCell},
    ffi::c_void,
    ptr,
    rc::Rc,
};

type CGImageSourceRef = *const c_void;
type CGImageRef = *const c_void;
type CGColorRef = *const c_void;
type CGColorSpaceRef = *const c_void;
type CGContextRef = *mut c_void;

#[link(name = "ImageIO", kind = "framework")]
unsafe extern "C" {
    fn CGImageSourceCreateWithURL(url: CFURLRef, options: CFDictionaryRef) -> CGImageSourceRef;
    fn CGImageSourceCopyPropertiesAtIndex(
        source: CGImageSourceRef,
        index: usize,
        options: CFDictionaryRef,
    ) -> CFDictionaryRef;
    fn CGImageSourceCreateThumbnailAtIndex(
        source: CGImageSourceRef,
        index: usize,
        options: CFDictionaryRef,
    ) -> CGImageRef;
    static kCGImagePropertyPixelWidth: CFStringRef;
    static kCGImagePropertyPixelHeight: CFStringRef;
    static kCGImageSourceCreateThumbnailFromImageAlways: CFStringRef;
    static kCGImageSourceCreateThumbnailWithTransform: CFStringRef;
    static kCGImageSourceThumbnailMaxPixelSize: CFStringRef;
    static kCGImageSourceShouldCacheImmediately: CFStringRef;
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    static kCGColorSpaceSRGB: CFStringRef;
    fn CGColorSpaceCreateWithName(name: CFStringRef) -> CGColorSpaceRef;
    fn CGColorSpaceRelease(space: CGColorSpaceRef);
    fn CGBitmapContextCreate(
        data: *mut c_void,
        width: usize,
        height: usize,
        bits_per_component: usize,
        bytes_per_row: usize,
        space: CGColorSpaceRef,
        bitmap_info: u32,
    ) -> CGContextRef;
    fn CGContextSetInterpolationQuality(context: CGContextRef, quality: i32);
    fn CGContextDrawImage(context: CGContextRef, rect: CGRect, image: CGImageRef);
    fn CGContextRelease(context: CGContextRef);
    fn CGImageGetBytesPerRow(image: CGImageRef) -> usize;
    fn CGImageGetWidth(image: CGImageRef) -> usize;
    fn CGImageGetHeight(image: CGImageRef) -> usize;
}

#[link(name = "QuartzCore", kind = "framework")]
unsafe extern "C" {
    static kCAFilterLinear: id;
}

#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {
    static NSWorkspaceDesktopImageScalingKey: id;
    static NSWorkspaceDesktopImageAllowClippingKey: id;
    static NSWorkspaceDesktopImageFillColorKey: id;
}

/// `kCGImageAlphaPremultipliedLast`: RGBA8, what [`average_color`] reads.
const ALPHA_PREMULTIPLIED_LAST: u32 = 1;
/// `kCGInterpolationHigh`.
const INTERPOLATION_HIGH: i32 = 3;
/// The side of the downscale the average colour is taken from.
const AVERAGE_SIDE: usize = 16;
/// Screens whose pictures are kept decoded.
const CACHED_SCREENS: usize = 3;
/// Pixels kept decoded, besides the newest picture, which is always kept. A
/// picture is at most `BLUR_DECODE_MAX` px on its long side (~1.3 MiB for a
/// 16:10 screen at 2x, ~2.3 MiB at the cap, 4 MiB for a square file).
const CACHED_BYTES: usize = 8 * 1024 * 1024;

/// A desktop picture decoded for one screen, shared by the cache and the
/// windows showing it.
pub(crate) struct CachedPicture {
    /// The picture at a fraction of the screen's resolution
    /// ([`decode_max_pixel_size`]); the layer scales it up.
    image: DecodedPicture,
    /// The file's size in pixels, from its header: the size the desktop's
    /// geometry is computed from.
    image_px: (f64, f64),
    /// `image`'s size in pixels.
    decoded_px: (f64, f64),
    /// How the desktop draws it on that screen.
    fit: Fit,
    /// The average colour of the picture.
    average: Option<Rgb>,
    /// What its pixels weigh.
    bytes: usize,
}

/// A decode's result: the picture, or `None` when it cannot be shown. Failures
/// are cached too, so a window does not try again on every fullscreen entry.
pub(crate) type Decoded = Option<Rc<CachedPicture>>;

/// Called on the main thread when the decode a window waits for lands.
pub(crate) type Waiter = Box<dyn FnOnce(&PictureKey, Decoded)>;

thread_local! {
    /// Main thread only: every use is from AppKit callbacks, `PlatformWindow`
    /// calls and foreground tasks.
    static PICTURES: RefCell<PictureCache<Decoded, Waiter>> =
        const { RefCell::new(PictureCache::new(CACHED_SCREENS, CACHED_BYTES)) };
    static MEMORY_PRESSURE: OnceCell<DispatchRetained<DispatchSource>> = const { OnceCell::new() };
}

/// The fullscreen half of a blurred window's background. Lives in the
/// window's state; every method runs on the main thread.
pub(crate) struct FullscreenBackdrop {
    /// `None` while the window is not a blurred window in native fullscreen.
    source: Option<BlurSource>,
    /// The picture this fullscreen window wants: its screen's, as the desktop
    /// shows it now. A decode for anything else is dropped.
    wanted: Option<PictureKey>,
    /// The picture under the blur, maybe a stale one while `wanted` decodes.
    shown: Option<Rc<CachedPicture>>,
    /// The layer-hosting view that draws `shown`. One +1 reference of our
    /// own, besides the content view's.
    picture_view: Option<id>,
    /// Whether the blur blends within the window (over the picture or the
    /// fill); `false` while it still blends behind it.
    within_window: bool,
    /// The window background under the blur while `within_window`.
    fill: Rgb,
    /// Between `windowWillEnterFullScreen:` and `windowDidEnterFullScreen:`.
    transitioning: bool,
}

/// A retained `CFURL`, immutable and so safe to use from any thread.
struct Url(CFURLRef);

// SAFETY: see the type's documentation; the reference is owned and released once.
unsafe impl Send for Url {}

impl Drop for Url {
    fn drop(&mut self) {
        // SAFETY: a +1 reference taken in `find_picture`.
        unsafe { CFRelease(self.0 as *const c_void) }
    }
}

/// An `ImageIO` image source, used by the one thread that opened it.
struct ImageSource(CGImageSourceRef);

impl Drop for ImageSource {
    fn drop(&mut self) {
        // SAFETY: a +1 reference from `CGImageSourceCreateWithURL`.
        unsafe { CFRelease(self.0) }
    }
}

/// A decoded desktop picture. `CGImage` is immutable and thread-safe.
struct DecodedPicture(CGImageRef);

// SAFETY: see the type's documentation; the reference is owned and released once.
unsafe impl Send for DecodedPicture {}

impl Drop for DecodedPicture {
    fn drop(&mut self) {
        // SAFETY: a +1 reference from `CGImageSourceCreateThumbnailAtIndex`.
        unsafe { CFRelease(self.0) }
    }
}

/// The off-main-thread half of showing a picture: the file is opened, sized
/// from its header, decoded at the size the screen draws it and averaged.
struct DecodeJob {
    url: Url,
    screen_px: (f64, f64),
    /// [`blur_long_side`] of the screen, as in the cache key.
    decode_long_side: u32,
    scaling: Option<i64>,
    allow_clipping: Option<bool>,
}

/// A screen's picture as the desktop shows it now: its cache key and the
/// decode that produces it. Found on the main thread without reading the file.
struct Wanted {
    key: PictureKey,
    job: DecodeJob,
}

impl DecodeJob {
    /// Decodes, downscales and averages the picture. Any thread.
    fn decode(self) -> Option<CachedPicture> {
        // SAFETY: the keys are ImageIO's own constants, the URL is a live
        // `CFURL` this job owns, and the options dictionary outlives the call.
        unsafe {
            let source = CGImageSourceCreateWithURL(self.url.0, ptr::null());
            if source.is_null() {
                return None;
            }
            let source = ImageSource(source);
            let image_px = pixel_size(source.0)?;
            let fit = desktop_fit(self.scaling, self.allow_clipping, image_px, self.screen_px);
            let max_pixel_size =
                decode_max_pixel_size(image_px, self.screen_px, fit, self.decode_long_side);
            if max_pixel_size == 0 {
                return None;
            }
            let key = |k: CFStringRef| CFString::wrap_under_get_rule(k);
            let options: CFDictionary<CFString, CFType> = CFDictionary::from_CFType_pairs(&[
                (
                    key(kCGImageSourceCreateThumbnailFromImageAlways),
                    CFBoolean::true_value().as_CFType(),
                ),
                (
                    key(kCGImageSourceCreateThumbnailWithTransform),
                    CFBoolean::true_value().as_CFType(),
                ),
                (
                    key(kCGImageSourceShouldCacheImmediately),
                    CFBoolean::true_value().as_CFType(),
                ),
                (
                    key(kCGImageSourceThumbnailMaxPixelSize),
                    CFNumber::from(max_pixel_size as i64).as_CFType(),
                ),
            ]);
            let image =
                CGImageSourceCreateThumbnailAtIndex(source.0, 0, options.as_concrete_TypeRef());
            if image.is_null() {
                return None;
            }
            let decoded = DecodedPicture(image);
            Some(CachedPicture {
                average: average_of(decoded.0),
                bytes: CGImageGetBytesPerRow(decoded.0) * CGImageGetHeight(decoded.0),
                decoded_px: (
                    CGImageGetWidth(decoded.0) as f64,
                    CGImageGetHeight(decoded.0) as f64,
                ),
                image_px,
                image: decoded,
                fit,
            })
        }
    }
}

/// The average colour of `image`, from a high-quality downscale to a few
/// pixels. Any thread.
unsafe fn average_of(image: CGImageRef) -> Option<Rgb> {
    unsafe {
        let mut pixels = vec![0u8; AVERAGE_SIDE * AVERAGE_SIDE * 4];
        let space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        if space.is_null() {
            return None;
        }
        let context = CGBitmapContextCreate(
            pixels.as_mut_ptr() as *mut c_void,
            AVERAGE_SIDE,
            AVERAGE_SIDE,
            8,
            AVERAGE_SIDE * 4,
            space,
            ALPHA_PREMULTIPLIED_LAST,
        );
        CGColorSpaceRelease(space);
        if context.is_null() {
            return None;
        }
        CGContextSetInterpolationQuality(context, INTERPOLATION_HIGH);
        let side = AVERAGE_SIDE as f64;
        CGContextDrawImage(
            context,
            CGRect::new(&CGPoint::new(0.0, 0.0), &CGSize::new(side, side)),
            image,
        );
        CGContextRelease(context);
        average_color(&pixels)
    }
}

/// The cached picture for `wanted`, or `None` while it decodes; `waiter`, if
/// any, is called when it lands (not on a hit). Starts the decode, off the main
/// thread, unless one for the same picture is already running.
fn request_picture(
    wanted: Wanted,
    background: &BackgroundExecutor,
    foreground: &ForegroundExecutor,
    waiter: Option<Waiter>,
) -> Option<Decoded> {
    watch_memory_pressure();
    let Wanted { key, job } = wanted;
    match PICTURES.with_borrow_mut(|cache| cache.request(&key, waiter)) {
        Lookup::Hit(decoded) => return Some(decoded),
        Lookup::Joined => return None,
        Lookup::Started => {}
    }
    let decode = background.spawn(async move { job.decode() });
    foreground
        .spawn(async move {
            let decoded = decode.await.map(Rc::new);
            if decoded.is_none() {
                log::info!(
                    "the desktop picture {} could not be decoded; fullscreen draws a flat background",
                    key.path
                );
            }
            let bytes = decoded.as_ref().map_or(0, |picture| picture.bytes);
            // The cache is released before any waiter runs: they lock their
            // window's state, and may ask the cache again.
            let waiters =
                PICTURES.with_borrow_mut(|cache| cache.finish(key.clone(), decoded.clone(), bytes));
            for waiter in waiters {
                waiter(&key, decoded.clone());
            }
        })
        .detach();
    None
}

/// Empties the cache when the system runs short of memory. Pictures shown by
/// a fullscreen window stay alive with it.
fn watch_memory_pressure() {
    extern "C" fn on_memory_pressure(_: *mut c_void) {
        PICTURES.with(|cache| {
            if let Ok(mut cache) = cache.try_borrow_mut() {
                cache.clear();
            }
        });
    }
    MEMORY_PRESSURE.with(|source| {
        source.get_or_init(|| {
            // SAFETY: a memory-pressure source on the main queue with a
            // context-free handler, resumed before it can ever be dropped
            // (destroying a suspended source is undefined behaviour); it lives
            // as long as the main thread.
            unsafe {
                let source = DispatchSource::new(
                    &raw const _dispatch_source_type_memorypressure as *mut _,
                    0,
                    // DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL
                    0x2 | 0x4,
                    Some(DispatchQueue::main()),
                );
                source.set_event_handler_f(on_memory_pressure);
                source.resume();
                source
            }
        });
    });
}

impl FullscreenBackdrop {
    pub(crate) fn new() -> Self {
        Self {
            source: None,
            wanted: None,
            shown: None,
            picture_view: None,
            within_window: false,
            fill: Rgb::new(0.0, 0.0, 0.0),
            transitioning: false,
        }
    }

    /// Whether the window draws its appearance: `false` only while a blurred
    /// fullscreen window has no picture to show and stands in with a flat fill.
    pub(crate) fn draws_appearance(&self) -> bool {
        self.source.is_none_or(BlurSource::draws_appearance)
    }

    /// Whether the window is a blurred window in native fullscreen.
    pub(crate) fn is_active(&self) -> bool {
        self.source.is_some()
    }

    /// Decodes the picture of a windowed blurred window's screen into the
    /// cache, off the main thread, so a fullscreen entry finds it there.
    ///
    /// # Safety
    /// Main thread; `window` is a live `NSWindow`.
    pub(crate) unsafe fn prefetch(
        window: id,
        background: &BackgroundExecutor,
        foreground: &ForegroundExecutor,
    ) {
        if let Some(wanted) = unsafe { find_picture(window) } {
            request_picture(wanted, background, foreground, None);
        }
    }

    /// Enters fullscreen (`transitioning`: from `windowWillEnterFullScreen:`),
    /// or refreshes the picture after a screen change: the screen's picture
    /// from the cache, installed at once with within-window blending. When it
    /// is not cached, the screen's previous picture stands in, if there is one;
    /// otherwise the blur keeps blending behind the window until the decode
    /// lands or [`Self::did_enter`]. `waiter` is how the decode reaches this
    /// window again ([`Self::picture_ready`]).
    ///
    /// # Safety
    /// Main thread; `window` is the live `NSWindow` and `blur_view` its blur
    /// view, a subview of the window's content view.
    pub(crate) unsafe fn enter(
        &mut self,
        window: id,
        blur_view: id,
        transitioning: bool,
        background: &BackgroundExecutor,
        foreground: &ForegroundExecutor,
        waiter: impl FnOnce() -> Waiter,
    ) {
        self.transitioning |= transitioning;
        unsafe {
            let Some(wanted) = find_picture(window) else {
                self.wanted = None;
                self.go_flat(window, blur_view);
                return;
            };
            let key = wanted.key.clone();
            self.wanted = Some(key.clone());
            self.source = Some(blur_source(true, true));
            match request_picture(wanted, background, foreground, Some(waiter())) {
                Some(decoded) => self.picture_ready(&key, decoded, window, blur_view),
                None => {
                    // A changed wallpaper or scale: what the screen showed last,
                    // until the new one lands. Otherwise a refresh keeps the
                    // picture it shows, and an entry keeps the blur as it is.
                    let stale = PICTURES.with_borrow(|cache| {
                        cache.latest_for_display(key.display, Option::is_some)
                    });
                    if let Some(Some(stale)) = stale {
                        self.show(stale, window, blur_view);
                    } else if !self.transitioning && !self.within_window {
                        // Already fullscreen (blurred there, or moved to a
                        // screen never seen): no animation to wait out, and the
                        // empty Space behind is black. The fill until it lands.
                        let fill = neutral(window);
                        without_animation(|| self.blend_within(window, blur_view, fill));
                    }
                }
            }
            if transitioning && self.shown.is_some() {
                // The picture went in above (cached, or the screen's stale
                // one). Commit it now rather than at the end of this run-loop
                // turn: AppKit snapshots the window and composites the first
                // frame of the fullscreen animation as soon as
                // `windowWillEnterFullScreen:` returns, and that frame must
                // already have the picture under the within-window blur.
                let _: () = msg_send![class!(CATransaction), flush];
            }
        }
    }

    /// The fullscreen transition is over. A blur still blending behind the
    /// window (its picture still decoding, or none to show) now blends within
    /// it over the fill, rather than over the black of the empty Space.
    ///
    /// # Safety
    /// As [`Self::enter`].
    pub(crate) unsafe fn did_enter(&mut self, window: id, blur_view: id) {
        self.transitioning = false;
        if self.is_active() && !self.within_window {
            unsafe {
                let fill = neutral(window);
                without_animation(|| self.blend_within(window, blur_view, fill));
            }
        }
    }

    /// Re-applies what [`Self::enter`] set on the window, after
    /// `set_background_appearance(Blurred)` reset the window's background in
    /// fullscreen. The picture is kept.
    ///
    /// # Safety
    /// As [`Self::enter`].
    pub(crate) unsafe fn reassert(&mut self, window: id, blur_view: id) {
        if self.within_window {
            let fill = self.fill;
            unsafe { without_animation(|| self.blend_within(window, blur_view, fill)) }
        }
    }

    /// Leaves fullscreen (or the blurred appearance): the picture is removed,
    /// a pending decode is dropped, and a blur view that remains blends behind
    /// the window again over `background`, the window's windowed background
    /// colour — all in one transaction.
    ///
    /// # Safety
    /// Main thread; `window` is the live `NSWindow`, `blur_view`, when given,
    /// its live blur view.
    pub(crate) unsafe fn leave(&mut self, window: id, blur_view: Option<id>, background: id) {
        self.source = None;
        self.wanted = None;
        self.shown = None;
        self.within_window = false;
        self.transitioning = false;
        unsafe {
            without_animation(|| {
                self.remove_picture();
                if let Some(blur_view) = blur_view {
                    NSVisualEffectView::setBlendingMode_(
                        blur_view,
                        NSVisualEffectBlendingMode::BehindWindow,
                    );
                    let _: () = msg_send![blur_view, setNeedsDisplay: YES];
                    window.setBackgroundColor_(background);
                }
            });
        }
    }

    /// The decode of `key` landed: shown if it is still the picture this
    /// window wants, else dropped. A failed decode leaves the neutral fill.
    ///
    /// # Safety
    /// As [`Self::enter`].
    pub(crate) unsafe fn picture_ready(
        &mut self,
        key: &PictureKey,
        decoded: Decoded,
        window: id,
        blur_view: id,
    ) {
        if !self.is_active() || self.wanted.as_ref() != Some(key) {
            return;
        }
        unsafe {
            match decoded {
                Some(picture) => self.show(picture, window, blur_view),
                None => self.go_flat(window, blur_view),
            }
        }
    }

    /// Puts `picture` under the blur, the blur within the window and the
    /// window background in the picture's colour, in one transaction.
    unsafe fn show(&mut self, picture: Rc<CachedPicture>, window: id, blur_view: id) {
        let same = self
            .shown
            .as_ref()
            .is_some_and(|shown| Rc::ptr_eq(shown, &picture));
        if same && self.within_window && self.picture_view.is_some() {
            return;
        }
        unsafe {
            let fill = picture.average.unwrap_or_else(|| neutral(window));
            without_animation(|| {
                let layer: id = match self.picture_view {
                    Some(view) => msg_send![view, layer],
                    None => self.add_picture_view(window, blur_view),
                };
                // The picture is decoded small and scaled up by the layer: the
                // gravity scales it to the layer's bounds (the screen's logical
                // size) for Fill, Fit and Stretch, and the contents scale keeps
                // a centred one at the file's size in screen pixels — the
                // desktop's geometry, computed from the file's size and the
                // screen's, not from the decoded pixels.
                let backing: f64 = msg_send![window, backingScaleFactor];
                let scale = picture_contents_scale(picture.image_px, picture.decoded_px, backing);
                let _: () = msg_send![layer, setContents: picture.image.0 as id];
                let _: () =
                    msg_send![layer, setContentsGravity: ns_string(picture.fit.contents_gravity())];
                let _: () = msg_send![layer, setContentsScale: scale];
                let _: () = msg_send![layer, setMagnificationFilter: kCAFilterLinear];
                let _: () = msg_send![layer, setMinificationFilter: kCAFilterLinear];
                let fill_color = fill_color(window).unwrap_or(ptr::null());
                let _: () = msg_send![layer, setBackgroundColor: fill_color];
                self.blend_within(window, blur_view, fill);
            });
        }
        self.shown = Some(picture);
    }

    /// No picture can be shown: the window stands in with the neutral fill
    /// and says so. Mid-transition, a blur still blending behind the window
    /// keeps doing so until [`Self::did_enter`].
    unsafe fn go_flat(&mut self, window: id, blur_view: id) {
        self.source = Some(blur_source(true, false));
        self.shown = None;
        unsafe {
            let fill = neutral(window);
            let blend = self.within_window || !self.transitioning;
            without_animation(|| {
                self.remove_picture();
                if blend {
                    self.blend_within(window, blur_view, fill);
                }
            });
        }
    }

    /// A new layer-hosting view below the blur view, returning its layer.
    unsafe fn add_picture_view(&mut self, window: id, blur_view: id) -> id {
        unsafe {
            let content_view: id = window.contentView();
            let view: id = msg_send![class!(NSView), alloc];
            let view: id = NSView::initWithFrame_(view, NSView::bounds(content_view));
            // Layer-hosting (the layer is set before `wantsLayer`), so AppKit
            // never draws into it or replaces its contents.
            let layer: id = msg_send![class!(CALayer), layer];
            let _: () = msg_send![layer, setMasksToBounds: YES];
            let _: () = msg_send![view, setLayer: layer];
            let _: () = msg_send![view, setWantsLayer: YES];
            view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable);
            let _: () = msg_send![
                content_view,
                addSubview: view
                positioned: NSWindowOrderingMode::NSWindowBelow
                relativeTo: blur_view
            ];
            self.picture_view = Some(view);
            layer
        }
    }

    /// Within-window blending over what is under the blur view, with `fill`
    /// as the window background beneath it all.
    unsafe fn blend_within(&mut self, window: id, blur_view: id, fill: Rgb) {
        unsafe {
            // Within-window blending samples the views under the blur view,
            // which have to be layer-backed (they are by default since 10.14).
            let content_view: id = window.contentView();
            let _: () = msg_send![content_view, setWantsLayer: YES];
            NSVisualEffectView::setBlendingMode_(
                blur_view,
                NSVisualEffectBlendingMode::WithinWindow,
            );
            let _: () = msg_send![blur_view, setNeedsDisplay: YES];
            window.setBackgroundColor_(NSColor::colorWithSRGBRed_green_blue_alpha_(
                nil, fill.r, fill.g, fill.b, 1.0,
            ));
        }
        self.within_window = true;
        self.fill = fill;
    }

    unsafe fn remove_picture(&mut self) {
        if let Some(view) = self.picture_view.take() {
            unsafe {
                let _: () = msg_send![view, removeFromSuperview];
                let _: () = msg_send![view, release];
            }
        }
    }
}

impl Drop for FullscreenBackdrop {
    fn drop(&mut self) {
        if let Some(view) = self.picture_view.take() {
            // SAFETY: our own +1 reference; the content view keeps its own
            // while the window lives.
            unsafe {
                let _: () = msg_send![view, release];
            }
        }
    }
}

/// Runs `f` in a `CATransaction` with implicit animations off, so whatever it
/// changes reaches the screen together, in one frame, without a fade.
unsafe fn without_animation<R>(f: impl FnOnce() -> R) -> R {
    unsafe {
        let _: () = msg_send![class!(CATransaction), begin];
        let _: () = msg_send![class!(CATransaction), setDisableActions: YES];
        let result = f();
        let _: () = msg_send![class!(CATransaction), commit];
        result
    }
}

/// The desktop picture of the window's screen, if it may be shown: a file
/// outside the privacy-guarded folders. Neither reads nor opens the file —
/// the decode does, off the main thread.
unsafe fn find_picture(window: id) -> Option<Wanted> {
    unsafe {
        let screen: id = msg_send![window, screen];
        if screen == nil {
            return None;
        }
        let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let url: id = msg_send![workspace, desktopImageURLForScreen: screen];
        if url == nil {
            return None;
        }
        let is_file: BOOL = msg_send![url, isFileURL];
        if is_file != YES {
            return None;
        }
        let path: id = msg_send![url, path];
        if path == nil {
            return None;
        }
        let path = path.to_str().to_owned();
        let home = std::env::var("HOME").ok();
        if !may_read_without_prompt(&path, home.as_deref()) {
            log::info!(
                "the desktop picture {path} is in a privacy-protected folder; fullscreen draws a flat background"
            );
            return None;
        }

        let frame: NSRect = NSScreen::frame(screen);
        let scale = NSScreen::backingScaleFactor(screen);
        let screen_px = (frame.size.width * scale, frame.size.height * scale);
        let options: id = msg_send![workspace, desktopImageOptionsForScreen: screen];
        let scaling = option(options, NSWorkspaceDesktopImageScalingKey).map(|n| {
            let n: NSInteger = msg_send![n, integerValue];
            n as i64
        });
        let allow_clipping = option(options, NSWorkspaceDesktopImageAllowClippingKey).map(|b| {
            let b: BOOL = msg_send![b, boolValue];
            b == YES
        });
        let fill = option(options, NSWorkspaceDesktopImageFillColorKey)
            .and_then(|color| srgb(color))
            .map(|(rgb, alpha)| rgb.to_rgba8(alpha));

        let screen_px_u32 = (screen_px.0.round() as u32, screen_px.1.round() as u32);
        let decode_long_side = blur_long_side(screen_px_u32);
        let key = PictureKey {
            display: display_id(screen),
            path,
            scaling,
            allow_clipping,
            fill,
            screen_px: screen_px_u32,
            decode_long_side,
        };
        CFRetain(url as *const c_void);
        Some(Wanted {
            key,
            job: DecodeJob {
                url: Url(url as CFURLRef),
                screen_px,
                decode_long_side,
                scaling,
                allow_clipping,
            },
        })
    }
}

/// The screen's `CGDirectDisplayID`.
unsafe fn display_id(screen: id) -> u32 {
    unsafe {
        let description: id = msg_send![screen, deviceDescription];
        let number: id = msg_send![description, objectForKey: ns_string("NSScreenNumber")];
        if number == nil {
            return 0;
        }
        msg_send![number, unsignedIntValue]
    }
}

/// The colour the desktop fills around a fitted or centred picture, if set.
unsafe fn fill_color(window: id) -> Option<CGColorRef> {
    unsafe {
        let screen: id = msg_send![window, screen];
        if screen == nil {
            return None;
        }
        let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let options: id = msg_send![workspace, desktopImageOptionsForScreen: screen];
        let color = option(options, NSWorkspaceDesktopImageFillColorKey)?;
        let cg: CGColorRef = msg_send![color, CGColor];
        (!cg.is_null()).then_some(cg)
    }
}

/// The fill for a window without a picture: its background colour in its
/// effective appearance, never lighter than a mid grey ([`neutral_fill`]).
unsafe fn neutral(window: id) -> Rgb {
    unsafe {
        let appearance: id = msg_send![window, effectiveAppearance];
        if appearance == nil {
            return neutral_fill(None, false);
        }
        let names =
            NSArray::arrayWithObjects(nil, &[NSAppearanceNameAqua, NSAppearanceNameDarkAqua]);
        let best: id = msg_send![appearance, bestMatchFromAppearancesWithNames: names];
        let dark = best != nil && {
            let equal: BOOL = msg_send![best, isEqualToString: NSAppearanceNameDarkAqua];
            equal == YES
        };
        neutral_fill(window_background_in(appearance), dark)
    }
}

/// `NSColor.windowBackgroundColor` resolved for `appearance` (macOS 11+).
unsafe fn window_background_in(appearance: id) -> Option<Rgb> {
    unsafe {
        let can: BOOL =
            msg_send![appearance, respondsToSelector: sel!(performAsCurrentDrawingAppearance:)];
        if can != YES {
            return None;
        }
        let resolved = Rc::new(Cell::new(None));
        let out = resolved.clone();
        let block = RcBlock::new(move || {
            let color: id = msg_send![class!(NSColor), windowBackgroundColor];
            out.set(srgb(color).map(|(rgb, _)| rgb));
        });
        let block: *const c_void = &*block as *const _ as *const c_void;
        let _: () = msg_send![appearance, performAsCurrentDrawingAppearance: block];
        resolved.get()
    }
}

/// An `NSColor` as sRGB components and alpha.
unsafe fn srgb(color: id) -> Option<(Rgb, f64)> {
    unsafe {
        let space: id = msg_send![class!(NSColorSpace), sRGBColorSpace];
        let color: id = msg_send![color, colorUsingColorSpace: space];
        if color == nil {
            return None;
        }
        let (mut r, mut g, mut b, mut a) = (0f64, 0f64, 0f64, 0f64);
        let _: () = msg_send![
            color,
            getRed: &mut r as *mut f64
            green: &mut g as *mut f64
            blue: &mut b as *mut f64
            alpha: &mut a as *mut f64
        ];
        Some((Rgb::new(r, g, b), a))
    }
}

unsafe fn option(options: id, key: id) -> Option<id> {
    if options == nil {
        return None;
    }
    unsafe {
        let value: id = msg_send![options, objectForKey: key];
        (value != nil).then_some(value)
    }
}

/// The picture's size in pixels, from its header.
unsafe fn pixel_size(source: CGImageSourceRef) -> Option<(f64, f64)> {
    unsafe {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, ptr::null());
        if properties.is_null() {
            return None;
        }
        let number = |key: CFStringRef| -> Option<f64> {
            let value = CFDictionaryGetValue(properties, key as *const c_void);
            if value.is_null() || CFGetTypeID(value) != CFNumberGetTypeID() {
                return None;
            }
            let mut out = 0f64;
            let ok = CFNumberGetValue(
                value as CFNumberRef,
                kCFNumberFloat64Type,
                &mut out as *mut f64 as *mut c_void,
            );
            ok.then_some(out)
        };
        let size = number(kCGImagePropertyPixelWidth).zip(number(kCGImagePropertyPixelHeight));
        CFRelease(properties as *const c_void);
        size
    }
}
