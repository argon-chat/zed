use std::{
    any::Any,
    collections::VecDeque,
    mem::ManuallyDrop,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use parking_lot::{Condvar, Mutex};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT},
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_11_0,
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
        System::Threading::{CreateEventW, WaitForSingleObject},
    },
    core::{IUnknown, Interface},
};

pub(super) type KeepAlive = Arc<dyn Any + Send + Sync>;

const MAX_REUSABLE_COMMAND_PAIRS: usize = 16;

#[derive(Clone, Copy, Default)]
pub(super) struct CommandStats {
    pub created: u64,
    pub reused: u64,
}

struct CommandPair {
    allocator: ID3D12CommandAllocator,
    commands: ID3D12GraphicsCommandList,
}

pub(crate) struct Device {
    pub(super) device: ID3D12Device,
    pub(super) queue: ID3D12CommandQueue,
    pub(super) adapter: IDXGIAdapter1,
    pub(super) dxgi_factory: IDXGIFactory6,
    video: Mutex<Weak<super::video::VideoBridge>>,
    retirement: Arc<Retirement>,
    pub(super) srv_descriptor_stride: u32,
}

struct Pending {
    value: u64,
    commands: Option<CommandPair>,
    _owners: Vec<KeepAlive>,
}

#[derive(Default)]
struct Submissions {
    next_value: u64,
    pending: VecDeque<Pending>,
    command_pool: Vec<CommandPair>,
    command_stats: CommandStats,
    worker_started: bool,
    shutdown: bool,
    failure: Option<String>,
    needs_signal: bool,
    retiring: usize,
}

struct Retirement {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    submissions: Mutex<Submissions>,
    changed: Condvar,
}

struct Event(HANDLE);

// An event handle names a process-wide kernel object and has exactly one owner.
unsafe impl Send for Event {}

impl Event {
    fn new() -> Result<Self> {
        Ok(Self(unsafe { CreateEventW(None, false, false, None)? }))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Err(error) = unsafe { CloseHandle(self.0) } {
            log::error!("D3D12 fence event close failed: {error}");
        }
    }
}

/// Set once a hardware adapter gave this process a device (`Device::new`).
static HAD_HARDWARE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Device {
    pub(super) fn new() -> Result<Arc<Self>> {
        let dxgi_factory: IDXGIFactory6 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))? };
        let wanted = std::env::var("VN_ADAPTER")
            .unwrap_or_default()
            .trim()
            .to_lowercase();
        let mut candidates = Vec::new();
        for index in 0.. {
            let adapter = match unsafe { dxgi_factory.EnumAdapters1(index) } {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => return Err(error).context("Enumerating D3D12 adapters"),
            };
            let descriptor = unsafe { adapter.GetDesc1()? };
            let end = descriptor
                .Description
                .iter()
                .position(|c| *c == 0)
                .unwrap_or(descriptor.Description.len());
            let name = String::from_utf16_lossy(&descriptor.Description[..end]);
            let software = descriptor.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
            candidates.push((adapter, name, software, descriptor.DedicatedVideoMemory));
        }
        let explicit_match = !wanted.is_empty()
            && candidates
                .iter()
                .any(|(_, name, _, _)| name.to_lowercase().contains(&wanted));
        if !wanted.is_empty() {
            if explicit_match {
                candidates.retain(|(_, name, _, _)| name.to_lowercase().contains(&wanted));
            } else {
                log::warn!(
                    "VN_ADAPTER={wanted:?} matches no D3D12 adapter; using automatic selection"
                );
            }
        }
        // Never silently replace GPU rendering with Microsoft's software
        // adapter. An explicit VN_ADAPTER remains useful for WARP fixtures.
        //
        // A machine that lists no hardware adapter at all (a VM without a GPU,
        // a basic-display RDP host) has nothing to replace, and the caller
        // panics without a device, so it gets the software adapter and a
        // warning. Not once this process has drawn on hardware: during a
        // driver update only the software adapter is listed for a few seconds,
        // and the device-loss recovery retries until the GPU is back.
        if !explicit_match {
            let hardware_listed = candidates.iter().any(|(_, _, software, _)| !software);
            if hardware_listed || HAD_HARDWARE.load(std::sync::atomic::Ordering::Relaxed) {
                candidates.retain(|(_, _, software, _)| !software);
            } else if !candidates.is_empty() {
                log::warn!(
                    "No hardware D3D12 adapter is listed; rendering with the software adapter"
                );
            }
        }
        // Keep the system's adapter order. Adapter 0 drives the primary display
        // and follows the user's per-app Windows graphics preference; on hybrid
        // laptops it is the integrated GPU unless high performance was chosen.
        // Ranking by dedicated memory would always wake the discrete GPU (and
        // copy every Present across adapters when the panel hangs off the iGPU).
        candidates.sort_by_key(|(_, _, software, _)| *software);
        let mut last_error = None;
        for (adapter, name, software, _) in candidates {
            let mut device: Option<ID3D12Device> = None;
            if let Err(error) =
                unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }
            {
                log::warn!("D3D12 adapter {name} unavailable: {error}");
                last_error = Some(error);
                continue;
            }
            let device = device.context("D3D12CreateDevice returned no device")?;
            log::info!("Using native D3D12 GPU: {name}");
            if !software {
                HAD_HARDWARE.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            return Self::from_native(device, adapter, dxgi_factory);
        }
        if let Some(error) = last_error {
            return Err(error).context("No adapter supports native Direct3D 12");
        }
        bail!("No DXGI adapters were available for native Direct3D 12")
    }

    fn from_native(
        device: ID3D12Device,
        adapter: IDXGIAdapter1,
        dxgi_factory: IDXGIFactory6,
    ) -> Result<Arc<Self>> {
        let queue: ID3D12CommandQueue = unsafe {
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?
        };
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let retirement = Arc::new(Retirement {
            device: device.clone(),
            queue: queue.clone(),
            fence,
            submissions: Mutex::new(Submissions::default()),
            changed: Condvar::new(),
        });
        let srv_descriptor_stride = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        Ok(Arc::new(Self {
            srv_descriptor_stride,
            device,
            queue,
            adapter,
            dxgi_factory,
            retirement,
            video: Mutex::new(Weak::new()),
        }))
    }

    pub(super) fn submit(
        &self,
        owners: Vec<KeepAlive>,
        record: impl FnOnce(&ID3D12GraphicsCommandList) -> Result<()>,
    ) -> Result<u64> {
        self.submit_after(owners, None, record)
    }

    /// Orders a producer queue's fence before this queue's commands without
    /// blocking the CPU. The producer must signal that fence independently.
    pub(super) fn submit_after(
        &self,
        mut owners: Vec<KeepAlive>,
        ready: Option<(&ID3D12Fence, u64)>,
        record: impl FnOnce(&ID3D12GraphicsCommandList) -> Result<()>,
    ) -> Result<u64> {
        unsafe { self.device.GetDeviceRemovedReason()? };
        let (pooled, retired) = {
            let mut submissions = self.retirement.submissions.lock();
            ensure!(!submissions.shutdown, "D3D12 device is shutting down");
            if let Some(error) = &submissions.failure {
                bail!("D3D12 submission failed previously: {error}");
            }
            self.retirement.start_worker(&mut submissions)?;
            let retired = drain_completed(&mut submissions, self.completed_value());
            (submissions.command_pool.pop(), retired)
        };
        // Recording a frame takes a while; only fence assignment, publication
        // and Execute/Signal need the mutex, which keeps values in queue order.
        let recorded = (|| {
            let reused = pooled.is_some();
            let pair = if let Some(pair) = pooled {
                unsafe {
                    pair.allocator.Reset()?;
                    pair.commands.Reset(&pair.allocator, None)?;
                }
                pair
            } else {
                let allocator: ID3D12CommandAllocator = unsafe {
                    self.device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?
                };
                let commands: ID3D12GraphicsCommandList = unsafe {
                    self.device.CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                        &allocator,
                        None,
                    )?
                };
                CommandPair {
                    allocator,
                    commands,
                }
            };
            record(&pair.commands)?;
            unsafe { pair.commands.Close()? };
            anyhow::Ok((pair, reused))
        })();
        let mut submissions = self.retirement.submissions.lock();
        let result = (|| {
            let (pair, reused) = recorded?;
            if reused {
                submissions.command_stats.reused += 1;
            } else {
                submissions.command_stats.created += 1;
            }
            ensure!(!submissions.shutdown, "D3D12 device is shutting down");
            if let Some(error) = &submissions.failure {
                bail!("D3D12 submission failed previously: {error}");
            }
            let value = submissions
                .next_value
                .checked_add(1)
                .filter(|value| *value != u64::MAX)
                .context("D3D12 fence value exhausted")?;
            let executable: ID3D12CommandList = pair.commands.cast()?;
            if let Some((fence, ready_value)) = ready {
                unsafe { self.queue.Wait(fence, ready_value)? };
                owners.push(Arc::new(fence.clone()));
            }
            submissions.next_value = value;
            // Publication precedes Execute: a failed Signal must retain every
            // command allocator and resource that the GPU may already consume.
            submissions.pending.push_back(Pending {
                value,
                commands: Some(pair),
                _owners: std::mem::take(&mut owners),
            });
            unsafe { self.queue.ExecuteCommandLists(&[Some(executable)]) };
            let signal = unsafe { self.queue.Signal(&self.retirement.fence, value) };
            if let Err(error) = &signal {
                submissions.failure = Some(error.to_string());
                submissions.needs_signal = true;
            }
            self.retirement.changed.notify_one();
            signal.context("Signalling native D3D12 submission completion")?;
            Ok(value)
        })();
        drop(submissions);
        // Both success and early recording errors must drop producer owners
        // outside the submission mutex: an owner can release its own Device.
        self.retirement.retire(retired);
        result
    }

    /// Whether a submission reached the queue without its completion being
    /// signalled, or never completed. Commands the GPU may still be reading
    /// have no fence to wait on then, and every later submission is refused.
    pub(super) fn submission_failed(&self) -> bool {
        self.retirement.submissions.lock().failure.is_some()
    }

    pub(super) fn command_stats(&self) -> CommandStats {
        self.retirement.submissions.lock().command_stats
    }

    pub(super) fn completed_value(&self) -> u64 {
        unsafe { self.retirement.fence.GetCompletedValue() }
    }

    pub(super) fn completion_point(&self) -> (ID3D12Fence, u64) {
        let submissions = self.retirement.submissions.lock();
        (self.retirement.fence.clone(), submissions.next_value)
    }

    pub(super) fn existing_video_bridge(&self) -> Option<Arc<super::video::VideoBridge>> {
        self.video.lock().upgrade()
    }

    pub(super) fn video_bridge(&self) -> Result<Arc<super::video::VideoBridge>> {
        let mut bridge = self.video.lock();
        if let Some(bridge) = bridge.upgrade() {
            return Ok(bridge);
        }
        let created = Arc::new(super::video::VideoBridge::new(self)?);
        *bridge = Arc::downgrade(&created);
        Ok(created)
    }

    pub(super) fn wait_for(&self, value: u64) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(30);
        unsafe { self.device.GetDeviceRemovedReason()? };
        // The frame loop mostly waits on fences that already completed.
        let event = if self.completed_value() < value {
            let event = Event::new()?;
            unsafe { self.retirement.fence.SetEventOnCompletion(value, event.0)? };
            Some(event)
        } else {
            None
        };
        while let Some(event) = event.as_ref().filter(|_| self.completed_value() < value) {
            let result = unsafe { WaitForSingleObject(event.0, 100) };
            ensure!(
                result == WAIT_OBJECT_0 || result == WAIT_TIMEOUT,
                "Waiting for D3D12 fence failed: {result:?}"
            );
            unsafe { self.device.GetDeviceRemovedReason()? };
            ensure!(
                Instant::now() < deadline,
                "D3D12 fence {value} did not complete within 30 seconds"
            );
        }
        unsafe { self.device.GetDeviceRemovedReason()? };
        let retired = {
            let mut submissions = self.retirement.submissions.lock();
            drain_completed(&mut submissions, self.completed_value())
        };
        self.retirement.retire(retired);
        // A worker may have removed a completed batch from `pending` and still
        // be dropping its owners. ResizeBuffers requires those references gone.
        let mut submissions = self.retirement.submissions.lock();
        while submissions.retiring != 0 {
            self.retirement.changed.wait(&mut submissions);
        }
        Ok(())
    }

    pub(super) fn wait_idle(&self) -> Result<()> {
        let value = self.submit(Vec::new(), |_| Ok(()))?;
        self.wait_for(value)
    }

    pub(super) fn create_buffer(
        &self,
        size: u64,
        heap_type: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource> {
        ensure!(size != 0, "D3D12 buffer must not be empty");
        let descriptor = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        self.create_resource(&descriptor, heap_type, state)
    }

    pub(super) fn create_texture(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource> {
        self.create_texture_multisampled(width, height, format, flags, state, 1)
    }

    pub(super) fn create_texture_multisampled(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
        samples: u32,
    ) -> Result<ID3D12Resource> {
        ensure!(
            width != 0 && height != 0 && samples != 0,
            "D3D12 texture dimensions/sample count must be positive"
        );
        let descriptor = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: samples,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        self.create_resource(&descriptor, D3D12_HEAP_TYPE_DEFAULT, state)
    }

    fn create_resource(
        &self,
        descriptor: &D3D12_RESOURCE_DESC,
        heap_type: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource> {
        let properties = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let mut resource = None;
        unsafe {
            self.device.CreateCommittedResource(
                &properties,
                D3D12_HEAP_FLAG_NONE,
                descriptor,
                state,
                None,
                &mut resource,
            )?
        };
        resource.context("CreateCommittedResource returned no resource")
    }

    pub(super) fn upload_texture(
        &self,
        destination: &ID3D12Resource,
        origin: (u32, u32),
        size: (u32, u32),
        bytes_per_pixel: u32,
        bytes: &[u8],
        stable_state: D3D12_RESOURCE_STATES,
    ) -> Result<u64> {
        let descriptor = unsafe { destination.GetDesc() };
        self.validate_resource_device(destination)?;
        ensure!(
            descriptor.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
                && descriptor.SampleDesc.Count == 1,
            "Only single-sample 2D textures can receive atlas uploads"
        );
        ensure!(
            u64::from(origin.0) + u64::from(size.0) <= descriptor.Width
                && u64::from(origin.1) + u64::from(size.1) <= u64::from(descriptor.Height),
            "D3D12 texture upload exceeds destination bounds"
        );
        ensure!(
            size.0 != 0 && size.1 != 0 && bytes_per_pixel != 0,
            "D3D12 texture upload must not be empty"
        );
        let expected_bytes_per_pixel = match descriptor.Format {
            DXGI_FORMAT_R8_UNORM => 1,
            DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM => 4,
            format => bail!("Unsupported packed texture upload format: {format:?}"),
        };
        ensure!(
            bytes_per_pixel == expected_bytes_per_pixel,
            "Texture upload pixel size does not match destination format"
        );
        let row_bytes = u64::from(size.0)
            .checked_mul(u64::from(bytes_per_pixel))
            .context("Texture row size overflow")?;
        let packed_size = row_bytes
            .checked_mul(u64::from(size.1))
            .context("Texture upload size overflow")?;
        ensure!(
            packed_size <= bytes.len() as u64,
            "Texture upload requires {packed_size} bytes; only {} provided",
            bytes.len()
        );
        let row_pitch = row_bytes
            .checked_add(255)
            .context("Texture row pitch overflow")?
            & !255;
        ensure!(
            row_pitch <= u64::from(u32::MAX),
            "Texture row pitch exceeds D3D12 limit"
        );
        let buffer_size = row_pitch
            .checked_mul(u64::from(size.1))
            .context("Texture staging size overflow")?;
        let mapped_size =
            usize::try_from(buffer_size).context("Texture staging size exceeds address space")?;
        let upload = self.create_buffer(
            buffer_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut mapped = std::ptr::null_mut();
        let empty = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { upload.Map(0, Some(&empty), Some(&mut mapped))? };
        if mapped.is_null() {
            unsafe { upload.Unmap(0, Some(&empty)) };
            bail!("D3D12 upload buffer mapping returned a null pointer");
        }
        for row in 0..size.1 as usize {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(row * row_bytes as usize),
                    mapped.cast::<u8>().add(row * row_pitch as usize),
                    row_bytes as usize,
                )
            };
        }
        let written = D3D12_RANGE {
            Begin: 0,
            End: mapped_size,
        };
        unsafe { upload.Unmap(0, Some(&written)) };
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: descriptor.Format,
                Width: size.0,
                Height: size.1,
                Depth: 1,
                RowPitch: row_pitch as u32,
            },
        };
        self.submit(
            vec![Arc::new(upload.clone()), Arc::new(destination.clone())],
            |commands| {
                transition(
                    commands,
                    destination,
                    stable_state,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let target = TextureLocation::subresource(destination, 0);
                let source = TextureLocation::placed(&upload, footprint);
                unsafe {
                    commands.CopyTextureRegion(
                        target.raw(),
                        origin.0,
                        origin.1,
                        0,
                        source.raw(),
                        None,
                    )
                };
                transition(
                    commands,
                    destination,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    stable_state,
                );
                Ok(())
            },
        )
    }

    pub(super) fn copy_texture(
        &self,
        source: &ID3D12Resource,
        source_state: D3D12_RESOURCE_STATES,
        destination: &ID3D12Resource,
        origin: (u32, u32),
        size: (u32, u32),
        destination_state: D3D12_RESOURCE_STATES,
        owner: KeepAlive,
        ready: Option<(&ID3D12Fence, u64)>,
    ) -> Result<u64> {
        let source_descriptor = unsafe { source.GetDesc() };
        let destination_descriptor = unsafe { destination.GetDesc() };
        ensure!(
            source_descriptor.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
                && destination_descriptor.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            "D3D12 atlas copy requires 2D resources"
        );
        ensure!(
            source_descriptor.Format == destination_descriptor.Format
                && source_descriptor.SampleDesc.Count == 1
                && destination_descriptor.SampleDesc.Count == 1,
            "D3D12 atlas copy format/sample mismatch"
        );
        ensure!(
            source_descriptor.Width >= u64::from(size.0) && source_descriptor.Height >= size.1,
            "D3D12 atlas source is too small"
        );
        ensure!(
            u64::from(origin.0) + u64::from(size.0) <= destination_descriptor.Width
                && u64::from(origin.1) + u64::from(size.1)
                    <= u64::from(destination_descriptor.Height),
            "D3D12 atlas copy exceeds destination bounds"
        );
        ensure!(
            size.0 != 0 && size.1 != 0,
            "D3D12 atlas copy must not be empty"
        );
        self.validate_resource_device(source)?;
        self.validate_resource_device(destination)?;
        ensure!(
            source.as_raw() != destination.as_raw(),
            "D3D12 atlas copy cannot alias its source"
        );
        self.submit_after(
            vec![
                owner,
                Arc::new(source.clone()),
                Arc::new(destination.clone()),
            ],
            ready,
            |commands| {
                transition(
                    commands,
                    source,
                    source_state,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                transition(
                    commands,
                    destination,
                    destination_state,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let source_location = TextureLocation::subresource(source, 0);
                let target = TextureLocation::subresource(destination, 0);
                let region = D3D12_BOX {
                    left: 0,
                    top: 0,
                    front: 0,
                    right: size.0,
                    bottom: size.1,
                    back: 1,
                };
                unsafe {
                    commands.CopyTextureRegion(
                        target.raw(),
                        origin.0,
                        origin.1,
                        0,
                        source_location.raw(),
                        Some(&region),
                    )
                };
                transition(
                    commands,
                    source,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    source_state,
                );
                transition(
                    commands,
                    destination,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    destination_state,
                );
                Ok(())
            },
        )
    }

    pub(super) fn validate_resource_device(&self, resource: &ID3D12Resource) -> Result<()> {
        let mut owner: Option<ID3D12Device> = None;
        unsafe { resource.GetDevice(&mut owner)? };
        let owner = owner.context("Resource has no D3D12 device")?;
        ensure!(
            owner.cast::<IUnknown>()? == self.device.cast::<IUnknown>()?,
            "D3D12 resource belongs to another physical device"
        );
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let mut submissions = self.retirement.submissions.lock();
        submissions.shutdown = true;
        self.retirement.changed.notify_one();
    }
}

fn drain_completed(submissions: &mut Submissions, completed: u64) -> Vec<Pending> {
    let mut retired = Vec::new();
    while submissions
        .pending
        .front()
        .is_some_and(|pending| pending.value <= completed)
    {
        if let Some(mut pending) = submissions.pending.pop_front() {
            if !submissions.shutdown
                && submissions.failure.is_none()
                && submissions.command_pool.len() < MAX_REUSABLE_COMMAND_PAIRS
            {
                if let Some(pair) = pending.commands.take() {
                    submissions.command_pool.push(pair);
                }
            }
            retired.push(pending);
        }
    }
    submissions.retiring += retired.len();
    retired
}

impl Retirement {
    fn retire(&self, retired: Vec<Pending>) {
        let count = retired.len();
        if count == 0 {
            return;
        }
        drop(retired);
        let mut submissions = self.submissions.lock();
        submissions.retiring -= count;
        self.changed.notify_all();
    }

    fn start_worker(self: &Arc<Self>, submissions: &mut Submissions) -> Result<()> {
        if submissions.worker_started {
            return Ok(());
        }
        let event = Event::new().context("Creating D3D12 retirement event")?;
        let retirement = self.clone();
        let worker = std::thread::Builder::new()
            .name("gpui-dx12-retire".into())
            .stack_size(128 * 1024)
            .spawn(move || retirement.run(event))
            .context("Starting D3D12 retirement worker")?;
        drop(worker);
        submissions.worker_started = true;
        Ok(())
    }

    fn run(&self, event: Event) {
        let mut stalled_since = Instant::now();
        let mut last_completed = 0;
        loop {
            let completed = unsafe { self.fence.GetCompletedValue() };
            let (retired, target, needs_signal, finished) = {
                let mut submissions = self.submissions.lock();
                let retired = drain_completed(&mut submissions, completed);
                while submissions.pending.is_empty() && !submissions.shutdown {
                    // Drop owner Arcs before sleeping: they can release the last
                    // Device and request shutdown of this very retirement state.
                    if !retired.is_empty() {
                        break;
                    }
                    self.changed.wait(&mut submissions);
                    stalled_since = Instant::now();
                }
                let target = submissions.pending.front().map(|pending| pending.value);
                let needs_signal = submissions.needs_signal.then_some(submissions.next_value);
                submissions.needs_signal = false;
                (
                    retired,
                    target,
                    needs_signal,
                    submissions.shutdown && submissions.pending.is_empty(),
                )
            };
            self.retire(retired);
            if finished {
                return;
            }
            let Some(target) = target else { continue };
            if completed != last_completed {
                stalled_since = Instant::now();
                last_completed = completed;
            }
            if let Some(value) = needs_signal {
                if let Err(error) = unsafe { self.queue.Signal(&self.fence, value) } {
                    log::error!("D3D12 completion signal retry failed: {error}");
                }
            }
            if let Err(error) = unsafe { self.device.GetDeviceRemovedReason() } {
                log::error!("D3D12 device removed while retiring resources: {error}");
                let pending = std::mem::take(&mut self.submissions.lock().pending);
                drop(pending);
                return;
            }
            if stalled_since.elapsed() > Duration::from_secs(30) {
                log::error!(
                    "D3D12 GPU did not complete pending work; retaining in-flight owners instead of releasing resources still in use"
                );
                let pending = std::mem::take(&mut self.submissions.lock().pending);
                // A live device with an unsignalled queue is not a safe point
                // for destruction. The exceptional leak is bounded by failure:
                // future submit calls are rejected for this device generation.
                self.submissions.lock().failure = Some("GPU completion timed out".into());
                std::mem::forget(pending);
                return;
            }
            match unsafe { self.fence.SetEventOnCompletion(target, event.0) } {
                Ok(()) => {
                    let result = unsafe { WaitForSingleObject(event.0, 100) };
                    if result != WAIT_OBJECT_0 && result != WAIT_TIMEOUT {
                        log::error!("D3D12 retirement fence wait failed: {result:?}");
                        let mut submissions = self.submissions.lock();
                        self.changed
                            .wait_for(&mut submissions, Duration::from_millis(100));
                    }
                }
                Err(error) => {
                    log::error!("D3D12 retirement event registration failed: {error}");
                    let mut submissions = self.submissions.lock();
                    self.changed
                        .wait_for(&mut submissions, Duration::from_millis(100));
                }
            }
        }
    }
}

pub(super) fn transition(
    commands: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    if before == after {
        return;
    }
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe {
        commands.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

/// Moves every resource between the same two states in one `ResourceBarrier`
/// call, so the GPU drains once rather than once per resource.
pub(super) fn transitions(
    commands: &ID3D12GraphicsCommandList,
    resources: &[&ID3D12Resource],
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    if before == after || resources.is_empty() {
        return;
    }
    let mut barriers: Vec<_> = resources
        .iter()
        .map(|resource| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some((*resource).clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        })
        .collect();
    unsafe {
        commands.ResourceBarrier(&barriers);
        for barrier in &mut barriers {
            ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
        }
    }
}

pub(super) struct TextureLocation(D3D12_TEXTURE_COPY_LOCATION);

impl TextureLocation {
    pub(super) fn subresource(resource: &ID3D12Resource, index: u32) -> Self {
        Self(D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(resource.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index,
            },
        })
    }

    pub(super) fn placed(
        resource: &ID3D12Resource,
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    ) -> Self {
        Self(D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(resource.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        })
    }

    pub(super) fn raw(&self) -> &D3D12_TEXTURE_COPY_LOCATION {
        &self.0
    }
}

impl Drop for TextureLocation {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.0.pResource) };
    }
}

#[cfg(test)]
impl Device {
    pub(super) fn read_texture_for_test(
        &self,
        resource: &ID3D12Resource,
        origin: (u32, u32),
        size: (u32, u32),
        bytes_per_pixel: u32,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<Vec<u8>> {
        let row_bytes = size.0 as usize * bytes_per_pixel as usize;
        let row_pitch = (row_bytes + 255) & !255;
        let buffer_size = row_pitch * size.1 as usize;
        let readback = self.create_buffer(
            buffer_size as u64,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: unsafe { resource.GetDesc() }.Format,
                Width: size.0,
                Height: size.1,
                Depth: 1,
                RowPitch: row_pitch as u32,
            },
        };
        let value = self.submit(
            vec![Arc::new(readback.clone()), Arc::new(resource.clone())],
            |commands| {
                transition(commands, resource, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                let source = TextureLocation::subresource(resource, 0);
                let destination = TextureLocation::placed(&readback, footprint);
                let region = D3D12_BOX {
                    left: origin.0,
                    top: origin.1,
                    front: 0,
                    right: origin.0 + size.0,
                    bottom: origin.1 + size.1,
                    back: 1,
                };
                unsafe {
                    commands.CopyTextureRegion(
                        destination.raw(),
                        0,
                        0,
                        0,
                        source.raw(),
                        Some(&region),
                    );
                }
                transition(commands, resource, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
                Ok(())
            },
        )?;
        self.wait_for(value)?;
        let mut mapped = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: buffer_size,
        };
        unsafe {
            readback.Map(0, Some(&range), Some(&mut mapped))?;
        }
        if mapped.is_null() {
            unsafe {
                readback.Unmap(0, None);
            }
            bail!("Readback mapped a null buffer");
        }
        let mut bytes = Vec::with_capacity(row_bytes * size.1 as usize);
        for row in 0..size.1 as usize {
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(mapped.cast::<u8>().add(row * row_pitch), row_bytes)
            });
        }
        let empty = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe {
            readback.Unmap(0, Some(&empty));
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
    fn native_command_pairs_reuse_only_after_their_fence() -> Result<()> {
        let device = Device::new()?;
        let gate: ID3D12Fence = unsafe { device.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let _release_gate = ReleaseGate(gate.clone());
        let mut first = 0;
        device.submit_after(Vec::new(), Some((&gate, 1)), |list| {
            first = list.as_raw() as usize;
            Ok(())
        })?;
        let mut second = 0;
        let fence = device.submit(Vec::new(), |list| {
            second = list.as_raw() as usize;
            Ok(())
        })?;
        assert_ne!(first, second, "in-flight command lists must not be reset");
        let busy = device.command_stats();
        assert_eq!(busy.created, 2);
        assert_eq!(busy.reused, 0);
        unsafe { gate.Signal(1)? };
        device.wait_for(fence)?;
        let mut reused = 0;
        let fence = device.submit(Vec::new(), |list| {
            reused = list.as_raw() as usize;
            Ok(())
        })?;
        device.wait_for(fence)?;
        assert!([first, second].contains(&reused));
        let completed = device.command_stats();
        assert_eq!(
            completed.created, 2,
            "steady-state submit created a new command pair"
        );
        assert_eq!(completed.reused, 1);
        Ok(())
    }

    struct ReleaseGate(ID3D12Fence);
    impl Drop for ReleaseGate {
        fn drop(&mut self) {
            if let Err(error) = unsafe { self.0.Signal(1) } {
                log::error!("Releasing native D3D12 test gate: {error}");
            }
        }
    }

    struct RetirementMarker(Arc<(Mutex<bool>, Condvar)>);

    impl Drop for RetirementMarker {
        fn drop(&mut self) {
            let (retired, changed) = &*self.0;
            *retired.lock() = true;
            changed.notify_one();
        }
    }

    #[test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
    fn native_dx12_retires_owners_without_another_frame() -> Result<()> {
        let device = Device::new()?;
        ensure!(
            device.existing_video_bridge().is_none(),
            "normal UI initialized D3D11"
        );
        let gate: ID3D12Fence = unsafe { device.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let _release_gate = ReleaseGate(gate.clone());
        let signal = Arc::new((Mutex::new(false), Condvar::new()));
        let marker = Arc::new(RetirementMarker(signal.clone()));
        let weak = Arc::downgrade(&marker);
        device.submit_after(vec![marker], Some((&gate, 1)), |_| Ok(()))?;
        assert!(
            weak.upgrade().is_some(),
            "producer was freed before its GPU fence"
        );
        unsafe {
            gate.Signal(1)?;
        }
        let (retired, changed) = &*signal;
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut retired = retired.lock();
        while !*retired {
            let remaining = deadline.saturating_duration_since(Instant::now());
            ensure!(
                !remaining.is_zero(),
                "owner was retained while the UI was idle"
            );
            changed.wait_for(&mut retired, remaining);
        }
        assert!(weak.upgrade().is_none());
        ensure!(
            device.existing_video_bridge().is_none(),
            "GPU uploads initialized D3D11"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
    fn native_dx12_cross_queue_copy_waits_and_preserves_padded_rows() -> Result<()> {
        let device = Device::new()?;
        let producer: ID3D12CommandQueue = unsafe {
            device
                .device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?
        };
        let ready: ID3D12Fence = unsafe { device.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let gate: ID3D12Fence = unsafe { device.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let _release_gate = ReleaseGate(gate.clone());
        let source = device.create_texture(
            5,
            3,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let destination = device.create_texture(
            13,
            7,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let pattern: Vec<u8> = (0..60).map(|index| (index * 13 + 7) as u8).collect();
        let upload = device.create_buffer(
            256 * 3,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut mapped = std::ptr::null_mut();
        unsafe {
            upload.Map(0, None, Some(&mut mapped))?;
        }
        ensure!(!mapped.is_null(), "upload mapped null");
        for row in 0..3 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pattern.as_ptr().add(row * 20),
                    mapped.cast::<u8>().add(row * 256),
                    20,
                );
            }
        }
        unsafe {
            upload.Unmap(0, None);
        }
        let allocator: ID3D12CommandAllocator = unsafe {
            device
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?
        };
        let commands: ID3D12GraphicsCommandList = unsafe {
            device
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?
        };
        transition(
            &commands,
            &source,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let destination_location = TextureLocation::subresource(&source, 0);
        let upload_location = TextureLocation::placed(
            &upload,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    Width: 5,
                    Height: 3,
                    Depth: 1,
                    RowPitch: 256,
                },
            },
        );
        unsafe {
            commands.CopyTextureRegion(
                destination_location.raw(),
                0,
                0,
                0,
                upload_location.raw(),
                None,
            );
        }
        transition(
            &commands,
            &source,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe {
            commands.Close()?;
            producer.Wait(&gate, 1)?;
            producer.ExecuteCommandLists(&[Some(commands.cast()?)]);
            producer.Signal(&ready, 1)?;
        }
        let resources = Arc::new((producer, allocator, commands, upload));
        let weak = Arc::downgrade(&resources);
        let copy = device.copy_texture(
            &source,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            &destination,
            (3, 2),
            (5, 3),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            resources,
            Some((&ready, 1)),
        )?;
        ensure!(
            device.completed_value() < copy,
            "consumer ignored the blocked producer fence"
        );
        assert!(weak.upgrade().is_some());
        unsafe {
            gate.Signal(1)?;
        }
        let result = device.read_texture_for_test(
            &destination,
            (3, 2),
            (5, 3),
            4,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        assert_eq!(result, pattern);
        assert!(
            weak.upgrade().is_none(),
            "completed producer resources were not retired"
        );
        Ok(())
    }
}
