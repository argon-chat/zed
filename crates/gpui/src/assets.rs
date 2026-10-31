use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use smallvec::SmallVec;

use image::{Delay, Frame};
use std::{
    borrow::Cow,
    fmt,
    hash::Hash,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

/// A source of assets for this app to use.
pub trait AssetSource: 'static + Send + Sync {
    /// Load the given asset from the source path.
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;

    /// List the assets at the given path.
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

/// A unique identifier for the image cache
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub usize);

#[derive(PartialEq, Eq, Hash, Clone)]
#[expect(missing_docs)]
pub struct RenderImageParams {
    pub image_id: ImageId,
    pub frame_index: usize,
}

/// A cached and processed image, in BGRA format
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    data: SmallVec<[Frame; 1]>,
    /// vue-native: one frame of a moving picture (a video, an animation). The
    /// host's [`crate::ImageResampler`] is not asked about it: its next frame
    /// is a new image, and a placeholder whose atlas tile the GPU fills has no
    /// pixels worth reading.
    dynamic: bool,
    /// vue-native: the size the picture has, when its pixels are a smaller
    /// copy of it ([`Self::with_natural_size`]). Layout and `object-fit` read
    /// this, so an image an embedder swapped for the copy it is drawn from
    /// lays out and fits exactly as the full decode did.
    natural: Option<Size<DevicePixels>>,
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Create a new image from the given data.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: data.into(),
            dynamic: false,
            natural: None,
        }
    }

    /// vue-native: marks this single-frame image as a smaller copy of a picture
    /// of `natural` pixels. [`Self::size`] and [`Self::as_bytes`] stay the copy's;
    /// [`Self::natural_size`] (layout, `object-fit`) answers `natural`, so the copy
    /// can stand in for the full decode in the asset cache
    /// ([`crate::ImageSource::replace_cached`]).
    pub fn with_natural_size(mut self, natural: Size<DevicePixels>) -> Self {
        self.natural = Some(natural);
        self
    }

    /// vue-native: the picture's own size in pixels: the one
    /// [`Self::with_natural_size`] recorded, else the pixels' size.
    pub fn natural_size(&self, frame_index: usize) -> Size<DevicePixels> {
        self.natural.unwrap_or_else(|| self.size(frame_index))
    }

    /// vue-native: are these pixels a smaller copy of the picture
    /// ([`Self::with_natural_size`])?
    pub fn is_reduced(&self) -> bool {
        self.natural.is_some()
    }

    /// vue-native: marks this image as one frame of a moving picture, which
    /// [`crate::Window::paint_image`] draws as it is (see `dynamic`).
    pub fn dynamic(mut self) -> Self {
        self.dynamic = true;
        self
    }

    /// vue-native: whether [`Self::dynamic`] marked this image.
    pub fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    /// Convert this image into a byte slice.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        self.data
            .get(frame_index)
            .map(|frame| frame.buffer().as_raw().as_slice())
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        self.data
            .get(frame_index)
            .map(|frame| {
                let (width, height) = frame.buffer().dimensions();
                size(width.into(), height.into())
            })
            .unwrap_or_default()
    }

    /// Get the size of this image, in pixels for display, adjusted for the scale factor.
    pub(crate) fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.natural_size(frame_index)
            .map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// Get the delay of this frame from the previous
    pub fn delay(&self, frame_index: usize) -> Delay {
        self.data
            .get(frame_index)
            .map(|frame| frame.delay())
            .unwrap_or(Delay::from_numer_denom_ms(100, 1))
    }

    /// Get the number of frames for this image.
    pub fn frame_count(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData")
            .field("id", &self.id)
            .field("size", &self.data.first().map(|f| f.buffer().dimensions()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    #[test]
    fn empty_render_image_does_not_panic() {
        let image = RenderImage::new(SmallVec::new());
        assert_eq!(image.frame_count(), 0);
        assert_eq!(image.size(0), Size::default());
        assert_eq!(image.as_bytes(0), None);
        assert_eq!(image.render_size(0), Size::default());
        assert_eq!(image.delay(0), Delay::from_numer_denom_ms(100, 1));
        let _ = format!("{image:?}");
    }

    #[test]
    fn a_frame_of_a_moving_picture_is_marked_dynamic() {
        let still = RenderImage::new(SmallVec::new());
        assert!(!still.is_dynamic());
        let frame = RenderImage::new(SmallVec::new()).dynamic();
        assert!(frame.is_dynamic());
        assert_ne!(still.id, frame.id);
    }

    #[test]
    fn a_reduced_copy_lays_out_at_the_natural_size() {
        let pixels = image::RgbaImage::new(4, 2);
        let copy = RenderImage::new(SmallVec::from_const([Frame::new(pixels)]))
            .with_natural_size(size(DevicePixels(400), DevicePixels(200)));
        assert!(copy.is_reduced());
        assert_eq!(copy.size(0), size(DevicePixels(4), DevicePixels(2)));
        assert_eq!(copy.natural_size(0), size(DevicePixels(400), DevicePixels(200)));
        assert_eq!(copy.render_size(0), size(Pixels::from(400.), Pixels::from(200.)));
        let full = RenderImage::new(SmallVec::from_const([Frame::new(image::RgbaImage::new(3, 3))]));
        assert!(!full.is_reduced());
        assert_eq!(full.natural_size(0), full.size(0));
    }
}
