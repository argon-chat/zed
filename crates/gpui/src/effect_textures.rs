//! vue-native: the images a `--shading` effect samples, its `SkTexture`
//! parameters.
//!
//! CSS names an image by URL; the embedder numbers each URL and resolves a
//! number to an image while painting ([`set_effect_texture_resolver`]). The
//! scene keeps one [`EffectTextureSet`] per distinct combination and an
//! [`EffectQuad`] names its set in the high bits of `flags`, so quads that
//! sample the same images still draw in one instanced call.

use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{Arc, RwLock, Weak},
};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{App, EffectQuad, EffectSpec, ImageId, RenderImage, Window};

/// `SkTexture` parameters an effect may declare. They bind at HLSL t3..t6,
/// WGSL `@group(2) @binding(3..6)` and Metal `texture(3..6)`, with the
/// repeating sampler at s2, `@binding(9)` and `sampler(2)`.
pub const EFFECT_TEXTURE_SLOTS: usize = 4;

/// Bits of [`EffectQuad::flags`] above this hold the quad's texture set, plus
/// one; zero samples no image.
pub const EFFECT_FLAG_TEXTURE_SET_SHIFT: u32 = 16;

/// One `SkTexture` parameter of an [`EffectSpec`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
pub struct EffectTextureParam {
    /// The image, numbered by the embedder; 0 when CSS gave none.
    pub source: u32,
    /// The first of the two parameter slots the painter fills with the
    /// image's width and height in texels, zero while it is missing.
    pub size_param: u8,
    /// Sampled as stored. Otherwise the renderer premultiplies it.
    pub linear: bool,
}

/// An image an effect quad samples.
#[derive(Clone, Debug)]
pub struct EffectTexture {
    /// Frame 0 is sampled.
    pub image: Arc<RenderImage>,
    /// See [`EffectTextureParam::linear`].
    pub linear: bool,
}

/// One quad's images, by binding.
pub type EffectTextureSet = [Option<EffectTexture>; EFFECT_TEXTURE_SLOTS];

/// Resolves [`EffectTextureParam::source`]: `None` while the image loads or
/// when it cannot, and the effect then reads its fallback.
pub type EffectTextureResolver = fn(u32, &mut Window, &mut App) -> Option<Arc<RenderImage>>;

static RESOLVER: RwLock<Option<EffectTextureResolver>> = RwLock::new(None);

/// Installs the embedder's resolver; without one every texture is missing.
pub fn set_effect_texture_resolver(resolver: EffectTextureResolver) {
    *RESOLVER.write().unwrap_or_else(|e| e.into_inner()) = Some(resolver);
}

/// The images `spec` names that are ready, or `None` when it names none.
pub(crate) fn resolve_effect_textures(
    spec: &EffectSpec,
    window: &mut Window,
    cx: &mut App,
) -> Option<Arc<EffectTextureSet>> {
    if spec
        .textures
        .iter()
        .all(|t| t.is_none_or(|t| t.source == 0))
    {
        return None;
    }
    let resolver = (*RESOLVER.read().unwrap_or_else(|e| e.into_inner()))?;
    let mut set = EffectTextureSet::default();
    for (slot, param) in spec.textures.iter().enumerate() {
        let Some(param) = param.filter(|p| p.source != 0) else {
            continue;
        };
        if let Some(image) = resolver(param.source, window, cx)
            && image.as_bytes(0).is_some()
        {
            set[slot] = Some(EffectTexture {
                image,
                linear: param.linear,
            });
        }
    }
    set.iter().any(Option::is_some).then(|| Arc::new(set))
}

/// Writes each image's size into its size slots: zero for a missing one.
pub(crate) fn write_effect_texture_sizes(
    spec: &EffectSpec,
    textures: Option<&EffectTextureSet>,
    params: &mut [f32],
) {
    for (slot, param) in spec.textures.iter().enumerate() {
        let Some(param) = param else { continue };
        let size = textures
            .and_then(|set| set[slot].as_ref())
            .map(|t| t.image.size(0))
            .map_or([0., 0.], |s| [s.width.0 as f32, s.height.0 as f32]);
        let at = param.size_param as usize;
        if let Some(target) = params.get_mut(at..at + 2) {
            target.copy_from_slice(&size);
        }
    }
}

pub(crate) type EffectTextureKey = [Option<(ImageId, bool)>; EFFECT_TEXTURE_SLOTS];

pub(crate) fn effect_texture_key(set: &EffectTextureSet) -> EffectTextureKey {
    std::array::from_fn(|i| set[i].as_ref().map(|t| (t.image.id, t.linear)))
}

impl EffectQuad {
    /// The index of this quad's set in [`crate::Scene::effect_texture_sets`].
    pub fn texture_set(&self) -> Option<usize> {
        (self.flags >> EFFECT_FLAG_TEXTURE_SET_SHIFT)
            .checked_sub(1)
            .map(|index| index as usize)
    }
}

impl EffectTexture {
    /// The BGRA8 texels to upload, premultiplied unless the texture is linear,
    /// with their width and height.
    pub fn texels(&self) -> Option<(u32, u32, Cow<'_, [u8]>)> {
        let bytes = self.image.as_bytes(0)?;
        let size = self.image.size(0);
        let (width, height) = (size.width.0 as u32, size.height.0 as u32);
        if width == 0 || height == 0 || bytes.len() < (width * height * 4) as usize {
            return None;
        }
        if self.linear || bytes.chunks_exact(4).all(|p| p[3] == 255) {
            return Some((width, height, Cow::Borrowed(bytes)));
        }
        let mut premultiplied = bytes.to_vec();
        for pixel in premultiplied.chunks_exact_mut(4) {
            let alpha = pixel[3] as u32;
            for channel in &mut pixel[..3] {
                *channel = ((*channel as u32 * alpha + 127) / 255) as u8;
            }
        }
        Some((width, height, Cow::Owned(premultiplied)))
    }
}

/// Frames a texture stays on the GPU unsampled before it is freed.
const EFFECT_TEXTURE_KEEP_FRAMES: u64 = 240;

/// A renderer's uploaded effect textures, keyed by image and colour handling.
pub struct EffectTextureCache<T> {
    entries: HashMap<(ImageId, bool), CachedEffectTexture<T>>,
    frame: u64,
}

struct CachedEffectTexture<T> {
    texture: Option<T>,
    image: Weak<RenderImage>,
    used: u64,
}

impl<T> Default for EffectTextureCache<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            frame: 0,
        }
    }
}

impl<T> EffectTextureCache<T> {
    /// Call once per drawn frame, before [`Self::get`]: frees what has gone
    /// unused for a while or whose image is gone.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.entries.retain(|_, entry| {
            frame - entry.used <= EFFECT_TEXTURE_KEEP_FRAMES && entry.image.strong_count() > 0
        });
    }

    /// The texture for `texture`, uploading it through `upload` (width,
    /// height, BGRA8 texels) when this image has none yet. An upload that
    /// fails is not retried while the image lives; its quads read black.
    pub fn get(
        &mut self,
        texture: &EffectTexture,
        upload: impl FnOnce(u32, u32, &[u8]) -> anyhow::Result<T>,
    ) -> Option<&T> {
        let key = (texture.image.id, texture.linear);
        let current = Arc::as_ptr(&texture.image);
        let stale = self
            .entries
            .get(&key)
            .is_none_or(|entry| entry.image.as_ptr() != current);
        if stale {
            let uploaded = texture.texels().and_then(|(width, height, texels)| {
                upload(width, height, &texels)
                    .inspect_err(|error| log::error!("uploading an effect texture: {error:#}"))
                    .ok()
            });
            self.entries.insert(
                key,
                CachedEffectTexture {
                    texture: uploaded,
                    image: Arc::downgrade(&texture.image),
                    used: self.frame,
                },
            );
        }
        let entry = self.entries.get_mut(&key)?;
        entry.used = self.frame;
        entry.texture.as_ref()
    }

    /// Frees every texture, as when the device is lost.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Frame, RgbaImage};

    fn image(pixels: &[[u8; 4]]) -> Arc<RenderImage> {
        let buffer = RgbaImage::from_raw(pixels.len() as u32, 1, pixels.concat()).unwrap();
        Arc::new(RenderImage::new(vec![Frame::new(buffer)]))
    }

    #[test]
    fn colour_texels_are_premultiplied_and_linear_ones_are_not() {
        let image = image(&[[200, 100, 50, 128], [10, 20, 30, 255]]);
        let colour = EffectTexture {
            image: image.clone(),
            linear: false,
        };
        let (_, _, texels) = colour.texels().unwrap();
        assert_eq!(&texels[..4], &[100, 50, 25, 128]);
        assert_eq!(&texels[4..], &[10, 20, 30, 255]);
        let data = EffectTexture {
            image,
            linear: true,
        };
        assert_eq!(&data.texels().unwrap().2[..4], &[200, 100, 50, 128]);
    }

    #[test]
    fn quads_sampling_different_images_draw_in_separate_batches() {
        use crate::{PrimitiveBatch, Scene, bounds, point, px, size};

        let set = |image: &Arc<RenderImage>| {
            let mut set = EffectTextureSet::default();
            set[0] = Some(EffectTexture {
                image: image.clone(),
                linear: false,
            });
            Some(Arc::new(set))
        };
        let (a, b) = (image(&[[0, 0, 0, 255]]), image(&[[9, 9, 9, 255]]));
        let quad = |x: f32| EffectQuad {
            effect_id: 3,
            bounds: bounds(point(px(x), px(0.)), size(px(10.), px(10.))).scale(1.),
            content_mask: crate::ContentMask {
                bounds: bounds(point(px(0.), px(0.)), size(px(500.), px(500.))).scale(1.),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut scene = Scene::default();
        for (x, textures) in [(0., set(&a)), (20., set(&b)), (40., set(&a)), (60., None)] {
            scene.insert_primitive(crate::Primitive::EffectQuad(quad(x), textures));
        }
        scene.finish();
        assert_eq!(
            scene.effect_texture_sets.len(),
            2,
            "one set per distinct image"
        );
        let batches: Vec<_> = scene
            .batches()
            .filter_map(|batch| match batch {
                PrimitiveBatch::EffectQuads { range, .. } => Some(range.len()),
                _ => None,
            })
            .collect();
        assert_eq!(batches, [1, 2, 1], "{batches:?}");

        let mut replayed = Scene::default();
        replayed.insert_primitive(crate::Primitive::EffectQuad(quad(0.), set(&b)));
        replayed.replay(0..scene.paint_operations.len(), &scene);
        replayed.finish();
        let sampled: Vec<_> = replayed
            .effect_quads
            .iter()
            .map(|quad| {
                quad.texture_set().map(|set| {
                    replayed.effect_texture_sets[set][0]
                        .as_ref()
                        .unwrap()
                        .image
                        .id
                })
            })
            .collect();
        assert!(sampled.contains(&Some(a.id)) && sampled.contains(&None));
        assert_eq!(sampled.iter().filter(|s| **s == Some(b.id)).count(), 2);
    }

    #[test]
    fn the_cache_uploads_an_image_once_and_again_when_it_is_replaced() {
        let mut cache = EffectTextureCache::<u32>::default();
        let mut uploads = 0;
        let first = EffectTexture {
            image: image(&[[0, 0, 0, 255]]),
            linear: false,
        };
        for _ in 0..3 {
            cache.begin_frame();
            cache.get(&first, |_, _, _| {
                uploads += 1;
                Ok(uploads)
            });
        }
        assert_eq!(uploads, 1);
        let next = EffectTexture {
            image: Arc::new(first.image.successor(vec![Frame::new(
                RgbaImage::from_raw(1, 1, vec![1, 2, 3, 255]).unwrap(),
            )])),
            linear: false,
        };
        cache.begin_frame();
        assert_eq!(cache.get(&next, |_, _, _| Ok(7)), Some(&7));
        drop(next);
        drop(first);
        cache.begin_frame();
        assert!(
            cache.entries.is_empty(),
            "a dropped image frees its texture"
        );
    }
}
