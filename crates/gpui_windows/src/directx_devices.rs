use anyhow::{Context, Result};
use gpui_util::ResultExt;
use itertools::Itertools;
use windows::Win32::{
    Foundation::HMODULE,
    Graphics::{
        Direct3D::{
            D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
        },
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
            D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS, D3D11_SDK_VERSION, D3D11CreateDevice,
            ID3D11Device, ID3D11DeviceContext, ID3D11Multithread,
        },
        Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG, DXGI_ADAPTER_FLAG_SOFTWARE,
            DXGI_CREATE_FACTORY_DEBUG, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory6,
        },
    },
};
use windows::core::Interface;

// ---------------------------------------------------------------------------
// vue-native: sharing the renderer's device
// ---------------------------------------------------------------------------
//
// Media Foundation decodes `<video>` straight onto THIS device
// (`IMFDXGIDeviceManager` + `IMFMediaEngine::TransferVideoFrame`), which is what
// keeps a decoded frame in VRAM instead of round-tripping through system memory.
// That needs three things this file is the only place able to give:
//
//   1. the device created with `D3D11_CREATE_DEVICE_VIDEO_SUPPORT` — without it
//      `ID3D11VideoDevice` cannot be obtained and the DXGI device manager
//      refuses it;
//   2. `ID3D11Multithread::SetMultithreadProtected(TRUE)` — MF's decoder issues
//      work on the immediate context from its own worker threads; Microsoft
//      documents this as required to avoid `GetDecoderBuffer` deadlocks;
//   3. a way OUT of this crate: `DirectXDevices` is `pub(crate)` and
//      `PlatformWindow` exposes only `sprite_atlas()` / `gpu_specs()`.
//
// (3) is published as a raw COM pointer rather than an `ID3D11Device` on
// purpose: the consumer crate compiles against a different `windows` crate
// version, and a COM pointer is an ABI, not a type. See
// [`renderer_d3d11_device`].

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

static SHARED_DEVICE: AtomicPtr<core::ffi::c_void> = AtomicPtr::new(std::ptr::null_mut());
static SHARED_DEVICE_GENERATION: AtomicU64 = AtomicU64::new(0);

/// The renderer's `ID3D11Device` as a **borrowed** raw COM pointer, plus a
/// generation that changes whenever the device is recreated (device-lost
/// recovery). `null` before the first window exists.
///
/// The pointer is NOT reference-counted for the caller. Wrap it with
/// `ID3D11Device::from_raw_borrowed(&ptr)` and `.clone()` it if you intend to
/// keep it, and re-check the generation each frame — after a device-lost the
/// old device is dead and anything built on it must be rebuilt.
///
/// Added for vue-native's `<video>` element (crates/vn-video); nothing in Zed
/// calls it.
pub fn renderer_d3d11_device() -> (*mut core::ffi::c_void, u64) {
    (
        SHARED_DEVICE.load(Ordering::Acquire),
        SHARED_DEVICE_GENERATION.load(Ordering::Acquire),
    )
}

fn publish_device(device: &ID3D11Device) {
    // Borrowed: `DirectXDevices` owns the reference for as long as the renderer
    // lives, and a consumer that keeps the pointer clones it into its own.
    SHARED_DEVICE.store(device.as_raw(), Ordering::Release);
    SHARED_DEVICE_GENERATION.fetch_add(1, Ordering::AcqRel);
}

pub(crate) fn try_to_recover_from_device_lost<T>(mut f: impl FnMut() -> Result<T>) -> Result<T> {
    (0..5)
        .map(|i| {
            if i > 0 {
                // Add a small delay before retrying
                std::thread::sleep(std::time::Duration::from_millis(100 + i * 10));
            }
            f()
        })
        .find_or_last(Result::is_ok)
        .unwrap()
        .context("DirectXRenderer failed to recover from lost device after multiple attempts")
}

#[derive(Clone)]
pub(crate) struct DirectXDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
}

impl DirectXDevices {
    pub(crate) fn new() -> Result<Self> {
        let debug_layer_available = check_debug_layer_available();
        let dxgi_factory =
            get_dxgi_factory(debug_layer_available).context("Creating DXGI factory")?;
        let (adapter, device, device_context, feature_level) =
            get_adapter(&dxgi_factory, debug_layer_available).context("Getting DXGI adapter")?;
        match feature_level {
            D3D_FEATURE_LEVEL_11_1 => {
                log::info!("Created device with Direct3D 11.1 feature level.")
            }
            D3D_FEATURE_LEVEL_11_0 => {
                log::info!("Created device with Direct3D 11.0 feature level.")
            }
            D3D_FEATURE_LEVEL_10_1 => {
                log::info!("Created device with Direct3D 10.1 feature level.")
            }
            _ => unreachable!(),
        }

        // vue-native: MF's decoder threads touch the immediate context we hand
        // them. Off by default — the probe in research/reports/m11/video.md §12
        // measured `previous = false` on this machine.
        if let Ok(mt) = device.cast::<ID3D11Multithread>() {
            let previous = unsafe { mt.SetMultithreadProtected(true) };
            log::info!(
                "D3D11 multithread protection enabled (was {}).",
                previous.as_bool()
            );
        } else {
            log::warn!(
                "ID3D11Multithread is unavailable on this device; sharing it with Media \
                 Foundation would be unsafe."
            );
        }
        publish_device(&device);

        Ok(Self {
            adapter,
            dxgi_factory,
            device,
            device_context,
        })
    }
}

/// One enumerated adapter, with the two facts the choice is made on.
struct Candidate {
    adapter: IDXGIAdapter1,
    name: String,
    /// `DedicatedVideoMemory` in bytes — the discriminator that actually works.
    /// A virtual display adapter (Parsec, SudoMaker, Spacedesk, a headless
    /// dongle emulator) is a real WDDM driver, is NOT flagged software, and
    /// reports 0 here; a discrete GPU reports its board's VRAM.
    dedicated_vram: u64,
    software: bool,
}

/// Every DXGI adapter, in enumeration order.
fn enumerate_adapters(dxgi_factory: &IDXGIFactory6) -> Vec<Candidate> {
    let mut out = Vec::new();
    for index in 0.. {
        let Ok(adapter) = (unsafe { dxgi_factory.EnumAdapters(index) }) else {
            break;
        };
        let Ok(adapter) = adapter.cast::<IDXGIAdapter1>() else {
            break;
        };
        let desc: DXGI_ADAPTER_DESC1 = unsafe { adapter.GetDesc1() }.unwrap_or_default();
        let name = String::from_utf16_lossy(&desc.Description)
            .trim_matches(char::from(0))
            .to_string();
        out.push(Candidate {
            adapter,
            name,
            dedicated_vram: desc.DedicatedVideoMemory as u64,
            software: DXGI_ADAPTER_FLAG(desc.Flags as i32).0 & DXGI_ADAPTER_FLAG_SOFTWARE.0 != 0,
        });
    }
    out
}

#[inline]
fn check_debug_layer_available() -> bool {
    #[cfg(debug_assertions)]
    {
        use windows::Win32::Graphics::Dxgi::{DXGIGetDebugInterface1, IDXGIInfoQueue};

        unsafe { DXGIGetDebugInterface1::<IDXGIInfoQueue>(0) }
            .log_err()
            .is_some()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

#[inline]
fn get_dxgi_factory(debug_layer_available: bool) -> Result<IDXGIFactory6> {
    let factory_flag = if debug_layer_available {
        DXGI_CREATE_FACTORY_DEBUG
    } else {
        #[cfg(debug_assertions)]
        log::warn!(
            "Failed to get DXGI debug interface. DirectX debugging features will be disabled."
        );
        DXGI_CREATE_FACTORY_FLAGS::default()
    };
    unsafe { Ok(CreateDXGIFactory2(factory_flag)?) }
}

#[inline]
fn get_adapter(
    dxgi_factory: &IDXGIFactory6,
    debug_layer_available: bool,
) -> Result<(
    IDXGIAdapter1,
    ID3D11Device,
    ID3D11DeviceContext,
    D3D_FEATURE_LEVEL,
)> {
    // vue-native: ORDER THE CANDIDATES, do not take enumeration order.
    //
    // The original loop took the first adapter that passed the feature test.
    // DXGI's enumeration order is not a preference order — on a machine with a
    // virtual display adapter installed (Parsec, SudoMaker, Spacedesk, a
    // headless-dongle emulator; this dev box enumerates two of them) the first
    // adapter is a software-ish display driver with no video-decode engine and
    // no dedicated VRAM. Rendering there is slow; worse, Media Foundation would
    // then decode `<video>` on the real GPU and every frame would cross adapters
    // through system memory, which is the entire cost the zero-copy design
    // exists to avoid (research/reports/m11/video.md §5, R1).
    //
    // Preference, in order:
    //   * `VN_ADAPTER=<substring>` — case-insensitive match on the adapter
    //     description. An explicit ask always wins, and says so if it misses.
    //   * not flagged `DXGI_ADAPTER_FLAG_SOFTWARE` (WARP).
    //   * most `DedicatedVideoMemory`.
    // Ties keep enumeration order, so a single-GPU machine behaves exactly as
    // it did before this change.
    let mut candidates = enumerate_adapters(dxgi_factory);
    for c in &candidates {
        log::info!(
            "DXGI adapter: {} — {} MiB dedicated VRAM{}",
            c.name,
            c.dedicated_vram / (1024 * 1024),
            if c.software { ", software" } else { "" }
        );
    }
    let wanted = std::env::var("VN_ADAPTER").unwrap_or_default().to_lowercase();
    if !wanted.trim().is_empty() {
        let wanted = wanted.trim();
        if candidates
            .iter()
            .any(|c| c.name.to_lowercase().contains(wanted))
        {
            candidates.retain(|c| c.name.to_lowercase().contains(wanted));
            log::info!("VN_ADAPTER={wanted:?} selected {}", candidates[0].name);
        } else {
            log::warn!(
                "VN_ADAPTER={wanted:?} matches no adapter on this machine; falling back to \
                 automatic selection."
            );
        }
    }
    // Stable sort: equal keys keep DXGI's order.
    candidates.sort_by_key(|c| (c.software, std::cmp::Reverse(c.dedicated_vram)));

    for candidate in &candidates {
        // Check to see whether the adapter supports Direct3D 11 and create
        // the device if it does.
        let mut context: Option<ID3D11DeviceContext> = None;
        let mut feature_level = D3D_FEATURE_LEVEL::default();
        if let Some(device) = get_device(
            &candidate.adapter,
            Some(&mut context),
            Some(&mut feature_level),
            debug_layer_available,
        )
        .log_err()
        {
            log::info!("Using GPU: {}", candidate.name);
            return Ok((
                candidate.adapter.clone(),
                device,
                context.unwrap(),
                feature_level,
            ));
        }
    }

    anyhow::bail!("No DXGI adapter supports the required Direct3D 11 feature set")
}

#[inline]
fn get_device(
    adapter: &IDXGIAdapter1,
    context: Option<*mut Option<ID3D11DeviceContext>>,
    feature_level: Option<*mut D3D_FEATURE_LEVEL>,
    debug_layer_available: bool,
) -> Result<ID3D11Device> {
    let mut device: Option<ID3D11Device> = None;
    // vue-native: `VIDEO_SUPPORT` is what lets `ID3D11VideoDevice` /
    // `IMFDXGIDeviceManager` be obtained from this device, i.e. what lets Media
    // Foundation decode `<video>` directly into textures the renderer can
    // sample. It costs nothing when no video is playing — the flag only makes
    // the video interfaces *available*.
    let device_flags = if debug_layer_available {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT
            | D3D11_CREATE_DEVICE_DEBUG
    } else {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT
    };
    unsafe {
        D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            device_flags,
            // 4x MSAA is required for Direct3D Feature Level 10.1 or better
            Some(&[
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0,
                D3D_FEATURE_LEVEL_10_1,
            ]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            feature_level,
            context,
        )?;
    }
    let device = device.unwrap();
    let mut data = D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS::default();
    unsafe {
        device
            .CheckFeatureSupport(
                D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
                &mut data as *mut _ as _,
                std::mem::size_of::<D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS>() as u32,
            )
            .context("Checking GPU device feature support")?;
    }
    if data
        .ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x
        .as_bool()
    {
        Ok(device)
    } else {
        Err(anyhow::anyhow!(
            "Required feature StructuredBuffer is not supported by GPU/driver"
        ))
    }
}
