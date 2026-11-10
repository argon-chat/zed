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
//! The picture is found and sized on the main thread (a file header read, no
//! decode) when fullscreen is entered, decoded off it at the screen's pixel
//! size, and shown when the decode arrives; the window's flat system
//! background stands in until then. A picture that cannot be shown — none, one
//! behind a privacy prompt ([`may_read_without_prompt`]), a dynamic or aerial
//! wallpaper ImageIO cannot read, a failed decode — leaves that flat
//! background, and the window reports its appearance as not drawn
//! (`PlatformWindow::is_background_appearance_drawn`), so the app can paint a
//! background of its own.
//!
//! Layer order inside the content view, bottom to top: the picture, the blur
//! view, gpui's view and its `CAMetalLayer`.

use crate::{
    NSStringExt,
    desktop_picture::{
        BlurSource, Fit, blur_source, decode_max_pixel_size, desktop_fit, may_read_without_prompt,
    },
    ns_string,
};
use cocoa::{
    appkit::{
        NSScreen, NSView, NSViewHeightSizable, NSViewWidthSizable, NSVisualEffectBlendingMode,
        NSVisualEffectView, NSWindow, NSWindowOrderingMode,
    },
    base::{id, nil},
    foundation::{NSInteger, NSRect},
};
use core_foundation::{
    base::{CFType, TCFType},
    boolean::CFBoolean,
    dictionary::CFDictionary,
    number::CFNumber,
    string::CFString,
};
use core_foundation_sys::{
    base::{CFGetTypeID, CFRelease},
    dictionary::{CFDictionaryGetValue, CFDictionaryRef},
    number::{CFNumberGetTypeID, CFNumberGetValue, CFNumberRef, kCFNumberFloat64Type},
    string::CFStringRef,
    url::CFURLRef,
};
use objc::{
    class, msg_send,
    runtime::{BOOL, YES},
    sel, sel_impl,
};
use std::{ffi::c_void, ptr};

type CGImageSourceRef = *const c_void;
type CGImageRef = *const c_void;
type CGColorRef = *const c_void;

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

#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {
    static NSWorkspaceDesktopImageScalingKey: id;
    static NSWorkspaceDesktopImageAllowClippingKey: id;
    static NSWorkspaceDesktopImageFillColorKey: id;
}

/// The fullscreen half of a blurred window's background. Lives in the
/// window's state; every method runs on the main thread.
pub(crate) struct FullscreenBackdrop {
    /// `None` while the window is not a blurred window in native fullscreen.
    source: Option<BlurSource>,
    /// The layer-hosting view that draws the picture. One +1 reference of our
    /// own, besides the content view's.
    picture_view: Option<id>,
    /// Bumped on every enter, refresh and leave, so a decode that lands after
    /// any of them is dropped.
    generation: u64,
    fit: Fit,
}

/// An `ImageIO` source opened on the main thread and decoded on another.
/// Image sources may move between threads; this one is used by one at a time.
struct ImageSource(CGImageSourceRef);

// SAFETY: see the type's documentation; the reference is owned and released once.
unsafe impl Send for ImageSource {}

impl Drop for ImageSource {
    fn drop(&mut self) {
        // SAFETY: a +1 reference from `CGImageSourceCreateWithURL`.
        unsafe { CFRelease(self.0) }
    }
}

/// A decoded desktop picture. `CGImage` is immutable and thread-safe.
pub(crate) struct DecodedPicture(CGImageRef);

// SAFETY: see the type's documentation; the reference is owned and released once.
unsafe impl Send for DecodedPicture {}

impl Drop for DecodedPicture {
    fn drop(&mut self) {
        // SAFETY: a +1 reference from `CGImageSourceCreateThumbnailAtIndex`.
        unsafe { CFRelease(self.0) }
    }
}

/// The off-main-thread half of showing a picture: decode it at the size it is
/// drawn, then hand the result to [`FullscreenBackdrop::show_decoded`] on the
/// main thread with [`Self::generation`].
pub(crate) struct DecodeJob {
    source: ImageSource,
    max_pixel_size: u32,
    pub(crate) generation: u64,
}

impl DecodeJob {
    /// Decodes and downscales the picture. Any thread.
    pub(crate) fn decode(self) -> Option<DecodedPicture> {
        // SAFETY: the keys are ImageIO's own constants, the source is a live
        // image source this job owns, and the options dictionary outlives the call.
        unsafe {
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
                    CFNumber::from(self.max_pixel_size as i64).as_CFType(),
                ),
            ]);
            let image = CGImageSourceCreateThumbnailAtIndex(
                self.source.0,
                0,
                options.as_concrete_TypeRef(),
            );
            (!image.is_null()).then_some(DecodedPicture(image))
        }
    }
}

impl FullscreenBackdrop {
    pub(crate) fn new() -> Self {
        Self {
            source: None,
            picture_view: None,
            generation: 0,
            fit: Fit::Fill,
        }
    }

    /// Whether the window draws its appearance: `false` only while a blurred
    /// fullscreen window stands in with its flat background.
    pub(crate) fn draws_appearance(&self) -> bool {
        self.source.is_none_or(BlurSource::draws_appearance)
    }

    /// Whether the window is a blurred window in native fullscreen.
    pub(crate) fn is_active(&self) -> bool {
        self.source.is_some()
    }

    /// Enters fullscreen, or refreshes the picture after a screen change:
    /// within-window blending, the flat system background under it, and the
    /// screen's picture found and sized. Returns the decode to run off the
    /// main thread, or `None` when the flat background is what is drawn.
    ///
    /// # Safety
    /// Main thread; `window` is the live `NSWindow` and `blur_view` its blur
    /// view, a subview of the window's content view.
    pub(crate) unsafe fn enter(&mut self, window: id, blur_view: id) -> Option<DecodeJob> {
        unsafe {
            self.generation += 1;
            self.blend(window, blur_view, NSVisualEffectBlendingMode::WithinWindow);
            let job = find_picture(window, self.generation);
            self.source = Some(blur_source(true, job.is_some()));
            match job {
                Some((job, fit)) => {
                    // A refresh keeps the shown picture until the new one arrives.
                    self.fit = fit;
                    Some(job)
                }
                None => {
                    self.remove_picture();
                    None
                }
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
        unsafe { self.blend(window, blur_view, NSVisualEffectBlendingMode::WithinWindow) }
    }

    /// Leaves fullscreen (or the blurred appearance): the picture is removed,
    /// a pending decode is dropped, and a blur view that remains blends behind
    /// the window again. The caller restores the window's background colour.
    ///
    /// # Safety
    /// Main thread; `blur_view`, when given, is the live blur view.
    pub(crate) unsafe fn leave(&mut self, blur_view: Option<id>) {
        unsafe {
            self.generation += 1;
            self.source = None;
            self.remove_picture();
            if let Some(blur_view) = blur_view {
                NSVisualEffectView::setBlendingMode_(
                    blur_view,
                    NSVisualEffectBlendingMode::BehindWindow,
                );
                let _: () = msg_send![blur_view, setNeedsDisplay: YES];
            }
        }
    }

    /// Shows a decoded picture under the blur view, replacing the one shown,
    /// or settles on the flat background when the decode failed. A result for
    /// an older [`DecodeJob`] than the latest is dropped.
    ///
    /// # Safety
    /// As [`Self::enter`].
    pub(crate) unsafe fn show_decoded(
        &mut self,
        generation: u64,
        picture: Option<DecodedPicture>,
        window: id,
        blur_view: id,
    ) {
        if generation != self.generation || self.source != Some(BlurSource::DesktopPicture) {
            return;
        }
        unsafe {
            let Some(picture) = picture else {
                log::info!(
                    "the desktop picture could not be decoded; fullscreen draws the window background"
                );
                self.source = Some(blur_source(true, false));
                self.remove_picture();
                return;
            };
            let content_view: id = window.contentView();
            let view: id = msg_send![class!(NSView), alloc];
            let view: id = NSView::initWithFrame_(view, NSView::bounds(content_view));
            // Layer-hosting (the layer is set before `wantsLayer`), so AppKit
            // never draws into it or replaces its contents.
            let layer: id = msg_send![class!(CALayer), layer];
            let scale: f64 = msg_send![window, backingScaleFactor];
            let _: () = msg_send![layer, setContents: picture.0 as id];
            let _: () =
                msg_send![layer, setContentsGravity: ns_string(self.fit.contents_gravity())];
            let _: () = msg_send![layer, setContentsScale: scale];
            let _: () = msg_send![layer, setMasksToBounds: YES];
            if let Some(fill) = fill_color(window) {
                let _: () = msg_send![layer, setBackgroundColor: fill];
            }
            let _: () = msg_send![view, setLayer: layer];
            let _: () = msg_send![view, setWantsLayer: YES];
            view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable);

            self.remove_picture();
            let _: () = msg_send![
                content_view,
                addSubview: view
                positioned: NSWindowOrderingMode::NSWindowBelow
                relativeTo: blur_view
            ];
            self.picture_view = Some(view);
        }
    }

    unsafe fn blend(&self, window: id, blur_view: id, mode: NSVisualEffectBlendingMode) {
        unsafe {
            // Within-window blending samples the views under the blur view,
            // which have to be layer-backed (they are by default since 10.14).
            let content_view: id = window.contentView();
            let _: () = msg_send![content_view, setWantsLayer: YES];
            NSVisualEffectView::setBlendingMode_(blur_view, mode);
            let _: () = msg_send![blur_view, setNeedsDisplay: YES];
            // What the blur blends with until (or instead of) the picture: the
            // system window background, never the black of an empty Space.
            let background: id = msg_send![class!(NSColor), windowBackgroundColor];
            window.setBackgroundColor_(background);
        }
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

/// The desktop picture of the window's screen, if it can be shown: a file
/// outside the privacy-guarded folders that ImageIO can read, with a usable
/// pixel size. Reads the file's header, never its pixels.
unsafe fn find_picture(window: id, generation: u64) -> Option<(DecodeJob, Fit)> {
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
        let path = path.to_str();
        let home = std::env::var("HOME").ok();
        if !may_read_without_prompt(path, home.as_deref()) {
            log::info!(
                "the desktop picture {path} is in a privacy-protected folder; fullscreen draws the window background"
            );
            return None;
        }
        let source = CGImageSourceCreateWithURL(url as CFURLRef, ptr::null());
        if source.is_null() {
            return None;
        }
        let source = ImageSource(source);
        let image = pixel_size(source.0)?;

        let frame: NSRect = NSScreen::frame(screen);
        let scale = NSScreen::backingScaleFactor(screen);
        let screen_px = (frame.size.width * scale, frame.size.height * scale);
        let options: id = msg_send![workspace, desktopImageOptionsForScreen: screen];
        let scaling = option(options, NSWorkspaceDesktopImageScalingKey).map(|n| {
            let n: NSInteger = msg_send![n, integerValue];
            n as i64
        });
        let clipping = option(options, NSWorkspaceDesktopImageAllowClippingKey).map(|b| {
            let b: BOOL = msg_send![b, boolValue];
            b == YES
        });
        let fit = desktop_fit(scaling, clipping, image, screen_px);
        let max_pixel_size = decode_max_pixel_size(image, screen_px, fit);
        if max_pixel_size == 0 {
            return None;
        }
        Some((
            DecodeJob {
                source,
                max_pixel_size,
                generation,
            },
            fit,
        ))
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
