#![cfg(target_os = "windows")]

mod clipboard;
mod destination_list;
mod dialog;
mod direct_manipulation;
mod direct_write;
#[cfg_attr(feature = "native-dx12", allow(dead_code))]
mod directx_atlas;
#[cfg_attr(feature = "native-dx12", allow(dead_code))]
mod directx_devices;
#[cfg_attr(feature = "native-dx12", allow(dead_code))]
mod directx_renderer;
#[cfg(feature = "native-dx12")]
mod directx12;
mod dispatcher;
mod display;
mod events;
mod keyboard;
mod platform;
mod system_notifications;
mod system_settings;
mod util;
mod vsync;
mod window;
mod wrapper;

pub(crate) use clipboard::*;
pub(crate) use destination_list::*;
pub(crate) use direct_write::*;
#[cfg(not(feature = "native-dx12"))]
pub(crate) use directx_devices::*;
#[cfg(not(feature = "native-dx12"))]
pub(crate) use directx_renderer::*;
#[cfg(feature = "native-dx12")]
pub(crate) use directx12::{DirectXDevices, DirectXRenderer};
#[cfg(feature = "native-dx12")]
pub(crate) use directx_devices::try_to_recover_from_device_lost;
#[cfg(feature = "native-dx12")]
pub(crate) use directx_renderer::DISABLE_DIRECT_COMPOSITION;
pub(crate) use dispatcher::*;
pub(crate) use display::*;
pub(crate) use events::*;
pub(crate) use keyboard::*;
pub(crate) use platform::*;
pub(crate) use system_notifications::*;
pub(crate) use system_settings::*;
pub(crate) use util::*;
pub(crate) use vsync::*;
pub(crate) use window::*;
pub(crate) use wrapper::*;

/// vue-native: the renderer's D3D11 device, so `<video>` can hand it to Media
/// Foundation and have decoded frames land in VRAM the renderer already owns.
/// See `directx_devices.rs`.
pub use directx_devices::{ExternalDirectXDevice, install_directx_device_factory};
#[cfg(not(feature = "native-dx12"))]
pub use directx_devices::renderer_d3d11_device;
#[cfg(feature = "native-dx12")]
pub use directx12::{RendererD3D11Handle, renderer_d3d11_device, ensure_renderer_d3d11_device};
#[cfg(feature = "native-dx12")]
pub use directx12::{RendererD3D12Handle, renderer_d3d12_device, install_renderer_d3d12_generation_observer};
pub use platform::WindowsPlatform;

pub(crate) use windows::Win32::Foundation::HWND;
