//! vue-native: an app's own `--shading` effects, as every renderer reads them. The CSS registry
//! (`sk-effects`) reads the parameter schemas from the same [`AppEffects::manifest`] and numbers the
//! effects the same way: `effect_id::COUNT + index`, in manifest order.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::bail;
use parking_lot::Mutex;

/// Names a directory of compiled app effects (`effects.json` and the shaders beside it).
/// A developer override: it wins over what the host installs with [`set_app_effects`].
pub const APP_EFFECTS_DIR_ENV: &str = "SK_EFFECTS_DIR";

/// The `effects.json` layout this build understands.
pub const APP_EFFECTS_SCHEMA: u32 = 9;

/// One generated shader of an [`AppEffect`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEffectSource {
    /// `effects.json` names no file for this target.
    Unlisted,
    /// `effects.json` names a file the app's effects do not carry.
    Missing(String),
    /// The file and its text.
    Present {
        /// As `effects.json` names it.
        file: String,
        /// The generated source.
        text: String,
    },
}

impl AppEffectSource {
    /// The file name and its text; `language` names the target in the error.
    pub fn get(&self, language: &str) -> anyhow::Result<(&str, &str)> {
        match self {
            AppEffectSource::Unlisted => {
                bail!("effects.json lists no {language} for it; rebuild the app with sk build")
            }
            AppEffectSource::Missing(file) => bail!("the app's effects have no {file}"),
            AppEffectSource::Present { file, text } => Ok((file, text)),
        }
    }
}

/// One app effect as far as a renderer is concerned; parameter names, defaults and ranges belong
/// to the CSS layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppEffect {
    /// The CSS function name.
    pub name: String,
    /// Decides the blend state and whether the renderer breaks the pass for a snapshot.
    pub needs_backdrop: bool,
    /// Whether it draws its element's painting (`[SkContent]`), and nothing without it.
    pub content: bool,
    /// Whether it keeps a history (`[SkHistory]`), which a second pipeline
    /// steps: blending off, a 16-bit float target.
    pub history: bool,
    /// For DirectX.
    pub hlsl: AppEffectSource,
    /// For wgpu.
    pub wgsl: AppEffectSource,
    /// For Metal.
    pub metal: AppEffectSource,
}

/// An app's effects: its `effects.json` and the generated shaders it names.
#[derive(Debug)]
pub struct AppEffects {
    /// Where they came from, for messages.
    pub origin: String,
    /// The `effects.json` text.
    pub manifest: String,
    effects: Vec<AppEffect>,
}

#[derive(serde::Deserialize)]
struct RawManifest {
    schema: u32,
    effects: Vec<RawEffect>,
}

#[derive(serde::Deserialize)]
struct RawEffect {
    name: String,
    backdrop: bool,
    #[serde(default)]
    content: bool,
    #[serde(default)]
    history: bool,
    #[serde(default)]
    hlsl: Option<String>,
    #[serde(default)]
    wgsl: Option<String>,
    #[serde(default)]
    metal: Option<String>,
}

impl AppEffects {
    /// `files` maps the names `manifest` gives the generated shaders to their source.
    pub fn new(
        origin: impl Into<String>,
        manifest: impl Into<String>,
        files: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Self, String> {
        let origin = origin.into();
        let manifest = manifest.into();
        let raw: RawManifest = serde_json::from_str(&manifest)
            .map_err(|e| format!("{origin}: effects.json is not readable ({e})"))?;
        if raw.schema != APP_EFFECTS_SCHEMA {
            return Err(format!(
                "{origin}: effects.json is schema {} but this build understands \
                 {APP_EFFECTS_SCHEMA} — rebuild the app with the sk that matches this host",
                raw.schema
            ));
        }
        let files: std::collections::HashMap<String, String> = files.into_iter().collect();
        let source = |file: Option<String>| match file {
            None => AppEffectSource::Unlisted,
            Some(file) => match files.get(&file).cloned() {
                Some(text) => AppEffectSource::Present { file, text },
                None => AppEffectSource::Missing(file),
            },
        };
        let effects = raw
            .effects
            .into_iter()
            .map(|effect| AppEffect {
                hlsl: source(effect.hlsl),
                wgsl: source(effect.wgsl),
                metal: source(effect.metal),
                needs_backdrop: effect.backdrop,
                content: effect.content,
                history: effect.history,
                name: effect.name,
            })
            .collect();
        Ok(Self {
            origin,
            manifest,
            effects,
        })
    }

    /// `<dir>/effects.json` and the `.hlsl`, `.wgsl` and `.metal` files beside it.
    pub fn from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let manifest_path = dir.join("effects.json");
        let manifest = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("{} could not be read ({e})", manifest_path.display()))?;
        let mut files = Vec::new();
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("{} could not be listed ({e})", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            let shader = path
                .extension()
                .is_some_and(|ext| ext == "hlsl" || ext == "wgsl" || ext == "metal");
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if shader {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{} could not be read ({e})", path.display()))?;
                files.push((name.to_string(), text));
            }
        }
        Self::new(dir.display().to_string(), manifest, files)
    }

    /// In manifest order: effect id `effect_id::COUNT + index`.
    pub fn effects(&self) -> &[AppEffect] {
        &self.effects
    }
}

struct Installed {
    read: bool,
    effects: Option<Arc<AppEffects>>,
}

static INSTALLED: Mutex<Installed> = Mutex::new(Installed {
    read: false,
    effects: None,
});
static GENERATION: AtomicU64 = AtomicU64::new(0);

fn app_effects_dir() -> Option<std::path::PathBuf> {
    std::env::var_os(APP_EFFECTS_DIR_ENV)
        .filter(|dir| !dir.is_empty())
        .map(std::path::PathBuf::from)
}

/// Hands the renderers and the CSS registry the app's effects (from its payload). Call it before
/// the first window; false when [`APP_EFFECTS_DIR_ENV`] is set, which wins, or when the effects were
/// already read.
pub fn set_app_effects(effects: AppEffects) -> bool {
    if app_effects_dir().is_some() {
        return false;
    }
    let mut installed = INSTALLED.lock();
    if installed.read {
        return false;
    }
    installed.read = true;
    installed.effects = Some(Arc::new(effects));
    true
}

/// Replaces the app's effects in a running app (a dev rebuild). Every renderer rebuilds the app's
/// pipelines on its next frame, and the CSS registry renumbers on its next read.
pub fn reload_app_effects(effects: AppEffects) {
    let mut installed = INSTALLED.lock();
    installed.read = true;
    installed.effects = Some(Arc::new(effects));
    GENERATION.fetch_add(1, Ordering::Release);
}

/// Moves on every [`reload_app_effects`]; a renderer compares it once per frame.
pub fn app_effects_generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// The app's effects: the last reload, else `$SK_EFFECTS_DIR` when set, else what the host
/// installed, else none.
pub fn app_effects() -> Option<Arc<AppEffects>> {
    let mut installed = INSTALLED.lock();
    if !installed.read {
        installed.read = true;
        installed.effects = app_effects_dir().and_then(|dir| {
            AppEffects::from_dir(&dir)
                .map_err(|error| {
                    let message = format!(
                        "{APP_EFFECTS_DIR_ENV} is set to `{}` but {error} — the app's own \
                         `--shading` effects are unavailable",
                        dir.display()
                    );
                    log::warn!("{message}");
                    eprintln!("skathi: {message}");
                })
                .ok()
                .map(Arc::new)
        });
    }
    installed.effects.clone()
}

/// A renderer's pipelines for the app's effects, `effect_id::COUNT + index`. [`Self::sync`]
/// rebuilds only the effects that changed since the last call; one whose new shader is unusable
/// keeps its previous pipeline, and only an effect that never built is a hole.
pub struct AppEffectPipelines<P> {
    generation: Option<u64>,
    slots: Vec<AppEffectSlot<P>>,
}

struct AppEffectSlot<P> {
    /// What the slot was last asked to build; `pipeline` may come from an earlier version.
    effect: AppEffect,
    pipeline: Option<P>,
}

impl<P> Default for AppEffectPipelines<P> {
    fn default() -> Self {
        Self {
            generation: None,
            slots: Vec::new(),
        }
    }
}

impl<P> AppEffectPipelines<P> {
    /// Catches up with [`reload_app_effects`]; false when nothing changed.
    pub fn sync(&mut self, mut build: impl FnMut(&AppEffect) -> anyhow::Result<P>) -> bool {
        let generation = app_effects_generation();
        if self.generation == Some(generation) {
            return false;
        }
        self.generation = Some(generation);
        let app = app_effects();
        let effects = app.as_deref().map(AppEffects::effects).unwrap_or_default();
        let mut previous = std::mem::take(&mut self.slots);
        for effect in effects {
            let old = previous
                .iter()
                .position(|slot| slot.effect.name == effect.name)
                .map(|index| previous.swap_remove(index));
            let pipeline = match old {
                Some(old) if old.effect == *effect => old.pipeline,
                old => match (build(effect), old.and_then(|old| old.pipeline)) {
                    (Ok(pipeline), _) => Some(pipeline),
                    (Err(error), Some(previous)) => {
                        advisory(
                            generation,
                            format!(
                                "`--shading: {}(…)` keeps its previous shader — the new one is \
                                 unusable: {error:#}",
                                effect.name
                            ),
                        );
                        Some(previous)
                    }
                    (Err(error), None) => {
                        advisory(
                            generation,
                            format!("`--shading: {}(…)` is unavailable — {error:#}", effect.name),
                        );
                        None
                    }
                },
            };
            self.slots.push(AppEffectSlot {
                effect: effect.clone(),
                pipeline,
            });
        }
        true
    }

    /// The pipeline of app effect `index`; None for a hole or past the end.
    pub fn get(&self, index: usize) -> Option<&P> {
        self.slots.get(index)?.pipeline.as_ref()
    }

    /// [`Self::get`], mutably.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut P> {
        self.slots.get_mut(index)?.pipeline.as_mut()
    }

    /// Every pipeline built, holes skipped.
    pub fn pipelines(&self) -> impl Iterator<Item = &P> {
        self.slots.iter().filter_map(|slot| slot.pipeline.as_ref())
    }
}

/// Once per message per reload: every window's renderer meets the same failure.
fn advisory(generation: u64, message: String) {
    static SEEN: Mutex<(u64, Vec<String>)> = Mutex::new((0, Vec::new()));
    let mut seen = SEEN.lock();
    if seen.0 != generation {
        *seen = (generation, Vec::new());
    }
    if seen.1.contains(&message) {
        return;
    }
    log::warn!("{message}");
    eprintln!("skathi: {message}");
    seen.1.push(message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_resolves_each_effect_to_its_shader_per_target() -> Result<(), String> {
        let app = AppEffects::new(
            "test",
            r#"{"schema": 9, "kind": "user", "effects": [
                {"name": "ripple", "backdrop": true, "hlsl": "ripple.hlsl", "wgsl": "ripple.wgsl"},
                {"name": "plain", "backdrop": false, "hlsl": "plain.hlsl"}
            ]}"#,
            [("ripple.wgsl".to_string(), "// ripple".to_string())],
        )?;
        let [ripple, plain] = app.effects() else {
            return Err(format!("{:?}", app.effects()));
        };
        assert!(ripple.needs_backdrop && !plain.needs_backdrop);
        assert_eq!(
            ripple.wgsl,
            AppEffectSource::Present {
                file: "ripple.wgsl".into(),
                text: "// ripple".into()
            }
        );
        assert_eq!(ripple.hlsl, AppEffectSource::Missing("ripple.hlsl".into()));
        assert_eq!(plain.wgsl, AppEffectSource::Unlisted);
        Ok(())
    }
}
