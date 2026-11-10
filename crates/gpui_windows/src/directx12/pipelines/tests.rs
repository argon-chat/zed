use super::*;
use windows::{Win32::Graphics::Direct3D::Fxc::D3DReflect, core::Interface};

fn modules() -> impl Iterator<Item = ShaderModule> {
    [
        ShaderModule::Quad,
        ShaderModule::Shadow,
        ShaderModule::BlurDownsample,
        ShaderModule::BlurUpsample,
        ShaderModule::BlurRect,
        ShaderModule::Underline,
        ShaderModule::PathRasterization,
        ShaderModule::PathSprite,
        ShaderModule::MonochromeSprite,
        ShaderModule::SubpixelSprite,
        ShaderModule::PolychromeSprite,
        ShaderModule::VnLayer,
    ]
    .into_iter()
    .chain(EffectShader::ALL.into_iter().map(ShaderModule::Effect))
}

fn contract(bytecode: &[u8], vertex: bool) -> Result<String> {
    ensure!(
        bytecode.starts_with(b"DXBC"),
        "native D3D12 shader must contain DXBC bytecode"
    );
    let mut raw = std::ptr::null_mut();
    unsafe {
        D3DReflect(
            bytecode.as_ptr().cast(),
            bytecode.len(),
            &ID3D12ShaderReflection::IID,
            &mut raw,
        )
    }?;
    let reflection = unsafe { ID3D12ShaderReflection::from_raw(raw) };
    let mut description = D3D12_SHADER_DESC::default();
    unsafe { reflection.GetDesc(&mut description) }?;
    ensure!(
        description.Version & 0xff == 0x51,
        "native shader must target model 5.1: {:#x}",
        description.Version
    );
    ensure!(
        description.Version >> 16 == u32::from(vertex),
        "native shader stage is incorrect"
    );
    let mut fields = vec![format!(
        "version={} flags={}",
        description.Version, description.Flags
    )];
    for index in 0..description.BoundResources {
        let mut resource = D3D12_SHADER_INPUT_BIND_DESC::default();
        unsafe { reflection.GetResourceBindingDesc(index, &mut resource) }?;
        let name = unsafe { resource.Name.to_string() }?;
        resource.Name = PCSTR::null();
        fields.push(format!("resource {name}: {resource:?}"));
    }
    for (input, count) in [
        (true, description.InputParameters),
        (false, description.OutputParameters),
    ] {
        for index in 0..count {
            let mut parameter = D3D12_SIGNATURE_PARAMETER_DESC::default();
            unsafe {
                if input {
                    reflection.GetInputParameterDesc(index, &mut parameter)
                } else {
                    reflection.GetOutputParameterDesc(index, &mut parameter)
                }
            }?;
            let name = unsafe { parameter.SemanticName.to_string() }?;
            parameter.SemanticName = PCSTR::null();
            fields.push(format!("signature input={input} {name}: {parameter:?}"));
        }
    }
    for index in 0..description.ConstantBuffers {
        let buffer = unsafe { reflection.GetConstantBufferByIndex(index) }
            .context("shader constant buffer missing")?;
        let mut buffer_description = D3D12_SHADER_BUFFER_DESC::default();
        unsafe { buffer.GetDesc(&mut buffer_description) }?;
        let name = unsafe { buffer_description.Name.to_string() }?;
        fields.push(format!(
            "buffer {name}: size={} type={:?} flags={}",
            buffer_description.Size, buffer_description.Type, buffer_description.uFlags
        ));
        for index in 0..buffer_description.Variables {
            let variable = unsafe { buffer.GetVariableByIndex(index) }
                .context("shader constant variable missing")?;
            let mut variable_description = D3D12_SHADER_VARIABLE_DESC::default();
            unsafe { variable.GetDesc(&mut variable_description) }?;
            let variable_name = unsafe { variable_description.Name.to_string() }?;
            fields.push(format!("variable {name}/{variable_name}: offset={} size={} flags={} textures={}:{} samplers={}:{}",
                variable_description.StartOffset, variable_description.Size, variable_description.uFlags,
                variable_description.StartTexture, variable_description.TextureSize,
                variable_description.StartSampler, variable_description.SamplerSize));
        }
    }
    fields.sort();
    Ok(fields.join("\n"))
}

#[test]
fn embedded_shader_contracts_match_current_sm51_hlsl() -> Result<()> {
    for module in modules() {
        let (expected_vertex, expected_fragment) =
            compile_module(module).with_context(|| format!("reference D3D12 shader {module:?}"))?;
        let (vertex, fragment) = embedded::module(module);
        for (artifact, reference, is_vertex) in [
            (vertex, expected_vertex.as_slice(), true),
            (fragment, expected_fragment.as_slice(), false),
        ] {
            assert_eq!(
                contract(artifact, is_vertex)?,
                contract(reference, is_vertex)?,
                "embedded {module:?} vertex={is_vertex} must preserve resource bindings, signatures and constant layouts"
            );
        }
    }
    Ok(())
}

#[test]
fn embedded_artifacts_track_current_sources_and_transitive_includes() -> Result<()> {
    for (path, expected_hash) in embedded::INPUTS {
        let source = std::fs::read(path).with_context(|| format!("shader build input {path}"))?;
        let hash = source.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        });
        assert_eq!(
            hash, *expected_hash,
            "embedded shader artifacts must be rebuilt after {path} changes"
        );
    }
    assert!(
        embedded::INPUTS
            .iter()
            .any(|(path, _)| path.ends_with("alpha_correction.hlsl")),
        "transitive shader includes must participate in rebuild tracking"
    );
    Ok(())
}

#[test]
#[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
fn native_prewarmed_pipelines_serve_their_first_use() -> Result<()> {
    let device = super::super::device::Device::new()?;
    let mut pipelines = Pipelines::new(&device.device)?;
    let format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;
    pipelines.prewarm(format);
    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    while pipelines.warmed.lock().len() < WARM_MODULES.len() {
        ensure!(Instant::now() < deadline, "pipeline warmup did not finish");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    for (module, samples) in WARM_MODULES {
        let warmed = pipelines
            .warmed
            .lock()
            .iter()
            .find(|cached| cached.key.kind == PipelineKind::Primitive(module) && cached.key.samples == samples)
            .map(|cached| cached.state.as_raw())
            .context("warmed pipeline missing")?;
        let served = pipelines.pipeline(module, format, samples)?;
        assert_eq!(served.as_raw(), warmed, "{module:?} must come from the warmup");
    }
    assert_eq!(pipelines.cache.len(), WARM_MODULES.len());
    Ok(())
}
