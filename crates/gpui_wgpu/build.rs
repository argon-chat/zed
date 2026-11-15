#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

use std::{env, error::Error, fmt::Write as _, fs, path::PathBuf};

/// `gpui::effect_id` order; `wgpu_renderer::backdrop` asserts the count.
const BUILTIN_EFFECTS: [&str; 3] = ["frost", "noise", "glow"];

/// Embeds the WGSL fragment stages `bun scripts/shaders.ts` checks in beside
/// the `.slang` sources. A layout without them still builds: the effects are
/// then unavailable at runtime, with an advisory.
fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SK_EFFECTS_GENERATED_DIR");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let generated = match env::var("SK_EFFECTS_GENERATED_DIR") {
        Ok(directory) => manifest_dir.join(directory),
        Err(_) => manifest_dir.join("../../../../crates/sk-effects/generated"),
    };

    let mut table = String::from(
        "pub(super) const BUILTIN_EFFECT_WGSL: [(&str, Option<&str>); BUILTIN_EFFECT_COUNT] = [\n",
    );
    for effect in BUILTIN_EFFECTS {
        let path = generated.join(format!("{effect}.wgsl"));
        println!("cargo:rerun-if-changed={}", path.display());
        if path.is_file() {
            let path = path.to_str().ok_or("effect shader path is not UTF-8")?;
            writeln!(table, "    ({effect:?}, Some(include_str!({path:?}))),")?;
        } else {
            println!(
                "cargo:warning=missing generated effect shader {} — `--shading: {effect}(…)` \
                 will be unavailable; run `bun run shaders` at the SDK root or point \
                 SK_EFFECTS_GENERATED_DIR at the directory",
                path.display()
            );
            writeln!(table, "    ({effect:?}, None),")?;
        }
    }
    table.push_str("];\n");
    writeln!(
        table,
        "pub(super) const BUILTIN_EFFECT_COUNT: usize = {};",
        BUILTIN_EFFECTS.len()
    )?;
    fs::write(
        PathBuf::from(env::var("OUT_DIR")?).join("builtin_effects.rs"),
        table,
    )?;
    Ok(())
}
