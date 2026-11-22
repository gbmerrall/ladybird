# WebGPU in Ladybird: a design proposal

## About this branch

`webgpu-wgpu-poc` is a proof of concept of WebGPU in Ladybird: wgpu running in the Compositor process,
following the design of WebGL. It is based on upstream commit `9932619ccb61` (2026-11-20) and is not rebased.

It is not a pull request. Ladybird's maintainers introduce all changes to the codebase themselves
(CONTRIBUTING.md), so this branch is evidence for the design discussion in issue 6514. It shows the approach works
and what it costs. A maintainer could adopt the approach or reimplement it. The branch was written with AI assistance
(see Documentation/CodePolicy.md).

**Status.** WebGPU works on Linux (Vulkan, tested on an NVIDIA RTX 3060, an AMD Radeon 780M and lavapipe) and on
macOS (Metal, tested on an Apple M2). The numbers are below.

**It is not robust yet.** An independent review of `9f0b575004` found that an ordinary web page can crash the
shared Compositor, in four ways, one of them with input the spec allows. It also found that ordinary `mapAsync`
calls can stall compositing for every tab. See **Known issues** below. The fixes are design-level, so the
proposal states them openly.

**Building on Linux.** Follow Documentation/BuildInstructionsLadybird.md. Nothing extra is needed: wgpu comes in
through Cargo (`Cargo.lock` is pinned, and `cargo fetch --locked` works). You need a Vulkan driver at run time.
lavapipe (Mesa's software Vulkan) is enough for the tests.

```
./Meta/ladybird.py build
./Meta/ladybird.py run                        # then open a WebGPU page
Build/release/bin/test-web --test-path Tests/LibWeb -f webgpu
cargo test -p webgpu_host -p webgpu_wire
```

**Building on macOS.** The same, with full Xcode installed (upstream's macOS build needs it) rather than only the
Command Line Tools. wgpu uses Metal. On an 8 GB machine, build with `./Meta/ladybird.py build -j4`.

**Switches.**
- `LADYBIRD_WEBGPU_READBACK_ONLY=1` turns off zero-copy presentation.
- `Meta/Linters/check_webgpu_seam.py` checks that only the wgpu backend names wgpu.

## The proposal

It is the issue's third approach: **wgpu as the first backend, behind a seam that Ladybird owns**, with WebGPU's
command execution in the Compositor process, as WebGL's already is.

### What Ladybird owns, and what it borrows

Most of a WebGPU implementation is browser work, not GPU work:
- the IDL and the JavaScript-facing objects
- promises and the event loop
- garbage collection of GPU objects
- the process boundary and its protocol
- the sandbox
- canvas presentation and compositing

On this branch all of that is Ladybird's own code, written the way the rest of LibWeb and the Compositor are.

Only one job is borrowed: validate a stream of WebGPU commands, compile WGSL, and drive Vulkan or Metal. That is the
part that takes years to get right. It is also the part where every engine is measured against the same
specification and test suite.

### The CTS is the contract, whichever engine sits behind it

Whatever executes the commands, Ladybird's WebGPU has to pass the WebGPU Conformance Test Suite (gpuweb/cts) and the
WPT tests built on it. That doesn't depend on the choice of engine. It is the bar for any implementation, including
one Ladybird writes itself.

The CTS's pass rate is therefore the measure of a backend:
- It is how this proof of concept was developed and judged: the CTS slices were run in headless Ladybird on two GPUs.
- It is how a replacement backend would be judged, by running the same queries on the same pages.
- It shows what a backend costs in conformance.

The engine is an implementation detail behind that contract. Choosing one now is not a permanent commitment.

### Swapping the backend

The seam is real for names, and shallow for semantics. For names:
- **The trait.** `trait Backend` has 46 methods, shaped after webgpu.h (the C API that both Dawn and wgpu-native
  implement). Each takes and returns `webgpu_wire` types, not wgpu types.
- **Where the backend is chosen.** In one place: `Server::new(Box<dyn Backend>, ...)`.
- **The wgpu backend.** About 2,500 lines in `src/wgpu_backend/`. It is the only code that changes when the
  engine does.
- **What doesn't change.**
  - the wire protocol
  - the IPC
  - LibWeb's bindings
  - canvas presentation (readback, and dma-buf zero copy, which goes through the trait as
    `import_canvas_texture`)
  - the tests
- **The tests.** The host's unit tests drive the server through wire commands, but they construct `WgpuBackend`
  directly. Running them against another backend means making that one helper take the backend as a parameter.
- **Enforcement.** The seam linter fails CI if wgpu, wgpu-hal, wgpu-types or naga are named anywhere else. Nothing
  outside the backend can come to depend on them by accident.

For semantics, the trait still carries wgpu's shape:
- **It is synchronous and blocking.** `map_buffer` returns the bytes, and `pop_error_scope` and
  `wait_for_submitted_work` return their results directly.
- **It encodes wgpu's model.** Error scopes, and device destruction (`destroy`, then a poll, then
  `is_destroyed_device`), are both wgpu's.
- **It carries Vulkan dma-buf import** under `cfg(unix)`.

A Dawn backend would fit behind most of the creation and encoding methods with little change. The mapping,
presentation and asynchronous parts would need the trait reshaped around completions (callbacks or futures, as
webgpu.h now has). That reshaping is also what H2 needs, so it is part of the work regardless of engine.

With that done, a Dawn backend would be a second implementation of the trait through webgpu.h. A Ladybird-native
engine, if one is ever wanted, would be a third. Either can be brought up next to the existing one, behind a switch,
and compared on the CTS before anything is removed. This is how Ladybird already treats Skia: a dependency behind
Ladybird's own painting interfaces.

### Why wgpu first

Issue 6514 has seen three directions: a WGSL compiler and Vulkan backend written from scratch, Dawn, and an
abstraction with Dawn behind it. A contributor's advice there was to settle on one engine rather than carry two.
wgpu is a good first one for these reasons:

- **It fits the build Ladybird already has.** LibWeb's bindings and the Compositor already contain Rust crates built
  by Cargo. wgpu is one more locked Cargo dependency:
  - no second build system
  - no extra vcpkg port
  - no third-party C++ toolchain requirements

  A clean release build of the host and all its dependencies takes about 16 seconds on a 16-core machine.
  Dawn brings a large C++ tree (Tint, Abseil, SPIRV-Tools and more), its own build expectations and a much longer
  compile.
- **It is a production WebGPU implementation.** wgpu is the engine under Firefox's WebGPU and under Deno.
  Its validation follows the specification and is exercised against the CTS upstream. Bugs found here can be
  fixed in wgpu, and they help other browsers too.
- **It covers Ladybird's platforms with native backends:** Vulkan on Linux, Metal on macOS, and D3D12 or Vulkan for
  Windows later. No translation layer is needed.
- **It is memory-safe where it matters.** The part that parses and validates untrusted shaders and command streams
  in a privileged process is written in Rust. That includes naga, which parses and validates WGSL. Memory safety
  is not crash safety, though. This branch drives wgpu's high-level Rust API, which panics on misuse because it is
  meant for trusted callers (C1). Firefox drives wgpu-core instead, whose calls return errors, behind its own
  validation. Ladybird would need to do the same, or validate everything first.
- **Its cost is small.** wgpu, naga and their dependencies add about 3.9 MB to the Compositor.
  The licences are all permissive (below).
- **Its known differences are small.** Content written against Chrome is developed on Dawn and Tint, so WGSL
  differences show up as shader errors. One three.js example hit one: naga can't constant-evaluate a `smoothstep`
  over abstract floats, which Tint can. Fixes for such gaps belong upstream in naga.

Writing an engine from scratch remains possible later. Draft PR #4958 shows the size of that job, and the CTS
bar above applies to it in full. The seam means Ladybird can start with wgpu now without ruling that out.

## Design

```
WebContent (sandboxed)                         Compositor (owns the GPU)
--------------------------------------------   ---------------------------------------------
LibWeb: WebGPU interfaces, in the Rust         Services/Compositor/WebGPU (Rust crate
bindings (Libraries/LibWeb/WebGPU/*.idl,       webgpu_host)
Bindings/Rust/src/web_gpu/)                       server.rs   decode, per-canvas state
  | records commands, allocates object ids        backend.rs  trait Backend   <-- the seam
  v                                               wgpu_backend/  the only code that names wgpu
LibWebGPU/Wire (crate webgpu_wire):                    |
commands, requests, replies, events                    v
  | batched per page, flushed each turn           wgpu 30 (Vulkan on Linux, Metal on macOS)
  v                                                    |
IPC: webgpu_commands / webgpu_events  ------->  CanvasHost: shows canvas frames (zero copy on
                                                Linux: canvases render into Skia's own images)
```

- **The seam.** `trait Backend` (`Services/Compositor/WebGPU/src/backend.rs`) is shaped like webgpu.h, so
  Dawn or Ladybird's own implementation could sit behind it. Only `src/wgpu_backend/` may name wgpu,
  wgpu-hal, wgpu-types or naga; `Meta/Linters/check_webgpu_seam.py`, run by `lint-ci.sh`, enforces it,
  and no manifest other than the host's may depend on them. LibWeb depends only on `webgpu_wire`.
- **The wire.** Ladybird's own protocol (`webgpu_wire`): object ids chosen by WebContent, WebGPU enums
  travelling as their spec strings, commands batched and flushed at the end of each event-loop turn,
  and asynchronous answers (adapters, devices, mapping, error scopes, compilation info, async
  pipelines) as request/reply events. A malformed stream is the sender misbehaving. The whole batch is decoded
  before any of it runs, so none of it runs, and the connection is dropped.
- **Lifetimes.** Wrappers' finalizers release ids with the next message; `destroy()` destroys. A
  destroyed device, and everything made on it, is let go at once. One gap: a child object doesn't yet keep its
  `GPUDevice` alive, as the spec requires (H4).
- **Presentation.** On Linux, a canvas on the compositor's GPU renders into images of Skia's device,
  shared as dma-bufs, with no CPU copy; elsewhere (other GPUs, lavapipe, macOS for now) the frame is
  read back asynchronously and uploaded. Presenting a frame doesn't wait for the GPU. Other calls still do (H2):
  `mapAsync`, unmapping after a write, `onSubmittedWorkDone`, device destruction and screenshots all wait on the
  Compositor's main thread.

### Technical notes

- **Why the Compositor.** It already owns the GPU for Skia and WebGL, and already has a sandbox profile that allows
  it. Putting WebGPU there keeps GPU access out of WebContent entirely. Frames can be shown without crossing
  another process boundary, and on Linux without leaving the GPU. A separate GPU process would be possible later,
  since the wire protocol doesn't care which process serves it.
- **No round trips for ordinary calls.** WebGPU is designed so that creating objects and recording commands never
  waits on the GPU process. WebContent picks object ids itself, so `createBuffer`, `createTexture`,
  `beginRenderPass` and the like return at once and are sent later in a batch. Only operations that are asynchronous
  in the spec cross back:
  - `requestAdapter` and `requestDevice`
  - `mapAsync`
  - `popErrorScope`
  - `getCompilationInfo`
  - `create*PipelineAsync`
  - `onSubmittedWorkDone`
- **Errors follow the spec's model, mostly.** An invalid call doesn't throw. The object becomes invalid on the
  device timeline, and the error reaches the page through error scopes or the `uncapturederror` event. A wgpu
  validation error maps directly onto that. Two caveats:
  - Errors the host raises itself skip error scopes, and a failed creation leaves no invalid object behind (M2).
  - Where wgpu's Rust API panics instead of returning an error, the Compositor aborts (C1).
- **Device loss.** `destroy()`, or a real loss, resolves `device.lost`. The host lets go of the device and every
  object made on it at once. Pages that create many devices, as the CTS does, therefore don't exhaust the driver.
- **Limits and fingerprinting.** Adapter limits are reported in coarse buckets, as the spec recommends. The host
  checks every requested limit and feature against the real adapter. Adapter info is still too detailed: it
  exposes the PCI device id and the full device name (M4).
- **Choosing the GPU.** With no power preference, the host prefers the adapter that Skia's compositor runs on. That
  is the condition for zero copy on machines with two GPUs.
- **Zero-copy synchronisation.**
  - wgpu to Skia: Skia only sees a frame once wgpu's submission has finished.
  - Skia to wgpu: an image is reused only after two newer frames have been shown and Skia no longer uses it. A
    sync-file semaphore would make this exact. Queue-family ownership transfers and agreed layouts are missing too
    (M6).
  - Fallback: any canvas the zero-copy path can't serve (another GPU, a format Skia can't share, storage usage) falls
    back to readback automatically.

## Status of the proof of concept

- **API breadth.** Buffers and mapping, textures, samplers, bind groups, render and compute pipelines (sync
  and async), render passes and bundles, compute passes, indirect draws, query sets (timestamps,
  occlusion), copies including `copyExternalImageToTexture`, error scopes, `uncapturederror`, device loss,
  compilation info, pipeline constants, labels and `wgslLanguageFeatures`.
- **webgpu-samples (54).** 48 of the 52 that report run without errors. The four that fail need a feature
  this GPU lacks, `importExternalTexture`, workers, or hit an unrelated LibWeb stream bug.
- **three.js r187.** 74 of 78 WebGPU examples run, at 60 fps: the 72 that need no assets, plus six with
  local glTF and HDR assets. Of the four that fail, three fail outside WebGPU and one hits a naga bug.
- **CTS.** About 79% of the 30k cases that run pass, across `api,operation`, `api,validation` and
  `web_platform`. The `idl` slice passes 121 of 130. These were run locally with a harness outside the tree, so
  they can't yet be reproduced from the branch.
- **Tests.** 10 LibWeb tests pass on the RTX 3060, the Radeon 780M, lavapipe and the M2. One is a screenshot
  Ref test. lavapipe is Mesa's software Vulkan and works in the sandbox, so CI needs no GPU. The host crate has
  13 unit tests and the wire crate 3. They prove the main paths work end to end. The gaps are listed under
  **Testing gaps** below.
- **macOS.** On an Apple M2 (Metal), the samples that run locally are clean except the same feature gaps. Presentation
  reads frames back there, and a 1920×1080 canvas still runs at 60 fps. The Compositor's sandbox profile needed no
  changes.
- **Performance.** Presenting a 3840×2160 canvas holds 60 fps with zero copy, against 43 fps read back,
  with a quarter of the Compositor's CPU time.
- **Size.** About 15,300 lines of code and tests over 149 files, not counting this document, the notes and the
  lockfiles.

## The dependency cost

wgpu 30.0.2 (pinned) brings 50 crates across Linux and macOS, all MIT, Apache-2.0 or Zlib
(28 MIT OR Apache-2.0, 8 Zlib OR Apache-2.0 OR MIT, 7 MIT, 2 Apache-2.0 OR MIT, 2 Apache-2.0, 1
MIT OR Apache-2.0 OR Zlib). Features are trimmed to `std`, `parking_lot` and `wgsl`, plus `vulkan` or
`metal` per platform. They add about 3.9 MB to the Compositor binary, and a clean release build of them takes about
16 seconds on a 16-core machine. Nothing in WebContent links them. This is the most likely objection, so the seam is
there to make wgpu replaceable rather than load-bearing.

## Security

- Untrusted WGSL and commands are validated by wgpu (naga for shaders) before they reach the driver. But wgpu's
  Rust API panics on some invalid input instead of returning an error, and naga can overflow the stack. The release
  build aborts on panic, so today a page can crash the Compositor (C1).
- The Compositor already runs with GPU access in a sandbox. On Linux, seccomp allows NVIDIA, RADV and
  lavapipe. On macOS, the existing profile already allows what Metal needs, because Skia and ANGLE use it
  too. This was confirmed on an M2.
- Limits are reported in coarse buckets, and every request is checked against the adapter's limits.
- What is missing is set out under **Known issues**:
  - crash safety (C1, H1)
  - not blocking the Compositor's main thread (H2)
  - limits on what one WebContent can use (H3)
  - exact zero-copy synchronisation (M6)

## Packaging

- `Meta/CMake/flatpak/cargo-sources.json` is regenerated, `cargo fetch --locked` works, and the host
  crate is a workspace member.
- Windows is untouched. wgpu's dx12 or Vulkan backend would be the path there.

## Open issues

- **Off the main thread.** The host must run on its own thread in the Compositor, or in its own process. Presenting
  doesn't block on the GPU, but mapping and several other calls do (H2).
- **Content-timeline validation.** Much of `api,validation` expects errors on the content timeline, in
  WebContent, which today are left to wgpu on the device timeline.
- **Missing pieces.**
  - `GPUPipelineError`
  - `importExternalTexture` (video)
  - WebGPU in workers
  - `requiredLimits` values of `undefined`
  - canvas readback for WebContent (`toDataURL`, `drawImage` of a WebGPU canvas)
- **macOS zero copy.** macOS reads frames back. Zero copy there would render into an IOSurface-backed Metal texture
  (`create_texture_from_hal`) and hand Skia the same IOSurface once the GPU is done.

## Known issues

These come from an independent review of `9f0b575004`. The reviewer read the code, and ran hand-written repro pages
in test-web on lavapipe plus extra host unit tests. Status:
- **confirmed:** reproduced
- **traced:** followed through the code, not run
- **suspected:** plausible, not proven

Severity:
- **critical:** an ordinary page can take down the shared Compositor
- **high:** a hostile WebContent can, or the damage is wide
- **medium:** spec or robustness defects

### Critical

**C1. An ordinary page can crash the Compositor.** Confirmed four ways on Linux (lavapipe). C1b was also reproduced
on macOS (Apple M2, Metal). There is one Compositor per browser, so every tab loses compositing. The browser restarts
the Compositor, but its restart budget of three is never refilled, and a fourth failure ends the browser. So the
damage outlasts the page.

The cause is that the wgpu backend drives wgpu's high-level Rust API with values that come straight from the page.
That API panics on misuse by design, because it is meant for trusted Rust callers. The release profile sets
`panic = "abort"`, so each panic kills the process.

- **a) Out-of-range buffer slices.** In `record_draw_op`, vertex and index buffers go through `Buffer::slice`,
  which panics when the range is outside the buffer. `setIndexBuffer(buffer, "uint16", 4096)` on a 16-byte buffer
  triggers it. The spec says that is a validation error.
- **b) Empty buffer slices, which the spec allows.** `setVertexBuffer(0, buffer, buffer.size)`, or an explicit size
  of 0, is valid. wgpu panics with "buffer slice can not be empty" (`size_expect_nonzero`, wgpu issue 3170).
  Correct content can hit this. Reproduced on both Linux and macOS.
- **c) Invalid render bundles.** wgpu's own backend treats every `RenderBundleEncoder::finish` error as fatal and
  panics. A bundle that draws with no pipeline set triggers it.
- **d) Deep WGSL expressions.** A chain of about 30,000 binary operators overflows the main thread's stack inside
  naga, and the process dies without a panic message. `catch_unwind` can't stop a stack overflow. Other deep
  nesting is rejected cleanly; this deserves a naga bug report.

### High

**H1. A hostile WebContent can crash the Compositor through the wire.** Confirmed in host unit tests. The page-side
code checks these values, so only a compromised WebContent can send them. That is exactly what a privileged process
has to survive.
- `map_buffer` and `unmap_buffer` slice buffers with ranges that come off the wire, without bounds checks.
- The canvas readback size is computed in `u32` and can overflow. In release builds it wraps, and a bogus frame
  size reaches the Compositor. A later length check happens to drop it.
- Every length in the wire (blobs, strings, vectors) is trusted downstream.

**H2. The Compositor's main thread waits on the GPU, with no timeout.** Confirmed: a 3.6 s stall on lavapipe. Each
of these waits for the GPU on the main thread that composites every tab:
- `mapAsync`
- unmapping after a write, which maps again
- `onSubmittedWorkDone`
- device destruction
- screenshots

`popErrorScope` and adapter and device requests block too. In the repro, a compute shader with a long loop followed
by `mapAsync` froze compositing for 3.6 s. With an unbounded loop and no GPU watchdog (lavapipe, for one), the stall
is unbounded. The `Backend` trait builds this in, since `map_buffer` returns the bytes. A watchdog that loses the
device would help, but the real fix is structural (below).

**H3. Decoding amplifies memory, and nothing limits one WebContent.** The amplification is confirmed. The rest is
traced.
- The wire decoder bounds an element count by the bytes left. That assumes at least one byte per element, but
  decoded elements are far larger (a `Command` is 216 bytes).
- The whole batch is decoded before any of it runs. In the repro, one 32 MiB message decoded to 805 MB of heap.
  IPC allows messages up to 4 GiB, so the Compositor's allocator can be made to abort.
- Nothing is limited per connection:
  - objects
  - devices: the driver's per-process device limit is shared by every tab
  - buffer and texture memory, in VRAM that Skia also uses
  - canvases
  - bulk data per message

**H4. A child object doesn't keep its device alive.** Traced.
- `GPUBuffer`, `GPUShaderModule` and `GPUCommandEncoder` hold only the device's id. The spec says each object
  holds its `[[device]]`, so the device should live while any child does.
- When the device wrapper is collected, the host forgets the device. wgpu keeps it alive through its children, so
  no memory is freed early.
- Every later lookup by device id fails, though. `mapAsync` on a surviving buffer then fails, and errors and loss
  events for that device are dropped.

### Medium

- **M1. Mapping is emulated.** The host maps, copies and unmaps at once, and WebContent only pretends the buffer is
  mapped. So GPU use of a buffer that is "mapped" isn't rejected, as the spec requires (confirmed). Unmapping after
  a write maps again and copies the whole range back. A failed mapping leaves the buffer unmappable from then on
  (confirmed). `mappedAtCreation` with a huge size aborts WebContent instead of throwing `RangeError`.
- **M2. Ladybird's own validation errors skip error scopes** (confirmed), and a failed creation leaves no invalid
  object, so each later use raises another error.
- **M3. The host's own GPU work runs on the page's device.** Readback, the clear pass and frame hand-off do. Their
  errors can land in the page's error scopes, or reach `uncapturederror` (traced).
- **M4. Adapter info exposes the PCI device id and the full device name.** Other engines report vendor and
  architecture only (traced).
- **M5. After a Compositor restart, WebGPU canvases stay blank.** The canvas context keeps its stale canvas id
  (traced).
- **M6. Zero-copy synchronisation is by convention.**
  - Skia to wgpu is ordered by frame count, not by a fence.
  - There are no queue-family ownership transfers.
  - Image layouts are agreed by convention.

  It works on the tested drivers. Under the Vulkan spec it is undefined, but the failure would be wrong pixels, not
  memory unsafety (suspected).
- **M7. Asynchronous pipeline creation compiles on the Compositor's main thread.** Script sees async behaviour, but
  large shaders stall compositing (traced).

### Low

- **L1. Id collisions.** A colliding id from a hostile WebContent can destroy another of that WebContent's own
  objects.
- **L2. Malformed streams.** A malformed stream drops the whole batch. This document said otherwise until this
  revision.
- **L3. Event order.** Uncaptured errors can arrive after replies to later requests in the same batch.
- **L4. Device destruction.** `device.destroy()` doesn't unmap buffers or reject pending `mapAsync` with
  `AbortError`. An adapter isn't consumed by its first device. A failed device request leaks an entry.
- **L5. Tests without a GPU.** The host unit tests pass without running when no Vulkan adapter is found, so CI
  without lavapipe would be green while proving nothing.

### What holds up

- Each WebContent connection has its own host, wgpu instance and id spaces. The review found no access across
  connections.
- Commands, requests and presentation stay in order on a connection.
- Replies settle in tasks, never re-entrantly.
- A WebContent's death drops all its GPU objects.

## Root causes and the fixes they need

The findings above share two root causes. Neither has a spot fix.

1. **wgpu's panicking high-level API is used on untrusted input, under `panic = "abort"`.** Checking ranges before
   each call fixes C1a-c and H1, but not the class. The design-level fix:
   - Drive wgpu-core, whose calls return errors, as Firefox does, or validate every input fully before it reaches
     wgpu.
   - Fuzz `Server::handle_commands`. A cargo-fuzz target would have found C1 and H1 quickly.
   - Run naga on a thread with a large stack, or cap shader nesting, for C1d.
2. **The Compositor's main thread waits on the GPU.** The fix:
   - An asynchronous host on its own thread or threads, polling without blocking and answering from completion
     callbacks.
   - A `Backend` trait reshaped around completions, which is also what a Dawn backend would need.
   - A watchdog that loses the device instead of stalling.

Two further pieces are needed:
- **Per-connection quotas** for objects, devices, memory, canvases and message sizes. Exceeding one loses the
  device, not the process. Decoding needs per-field caps, and should decode and execute as a stream.
- **Ideally, a separate failure domain**, such as a GPU process. wgpu, naga and the drivers' shader compilers would
  then run outside the process that composites every tab, so a crash or stall costs WebGPU, not the browser. If
  WebGPU stays in the Compositor, it needs at least its own thread with a large stack, and failures that lose the
  device rather than the process.

## Testing gaps

The tests cover the main paths. The review lists these gaps, most important first:

1. No tests of the server with hostile or malformed input. The wire's truncation test only checks that decoding
   doesn't panic. Fuzzing `handle_commands` is the most valuable single addition.
2. No edge cases that are valid by the spec: zero-size slices, empty buffers, ranges ending at the end of a
   buffer. C1b would have been caught.
3. No garbage-collection or lifetime tests, for example a device dropped while its children live (H4).
4. No tests of Compositor loss and recovery (M5), or of two WebContents at once.
5. Device-loss tests cover only `destroy()`. Pending maps at loss, and errors after loss, aren't covered.
6. Presentation has one Ref test. Resize, reconfiguring, and zero copy against readback aren't covered.
7. The CTS numbers can't be reproduced from the tree. There is no in-tree CTS slice.

## How the work divides

For a maintainer weighing the approach, the branch falls into parts that each stand on their own. This is a map of
the code, not a plan to submit it:

- **Rust bindings generator fixes WebGPU needed:**
  - namespace constants
  - `[Clamp]` on `unsigned short`
  - dictionaries with no members of their own

  These are useful without WebGPU.
- **`webgpu_wire`** (`Libraries/LibWebGPU/Wire`): the protocol, with unit tests.
- **`webgpu_host`** (`Services/Compositor/WebGPU`):
  - the Backend seam
  - the wgpu backend
  - the seam linter
  - unit tests

  It has no Ladybird dependencies of its own.
- **Compositor and IPC plumbing:**
  - `CanvasHost`
  - the IPC messages
  - batching in `CompositorHost`
  - readback presentation
- **LibWeb interfaces**, each area with tests:
  - adapter, device, errors and device loss
  - buffers and mapping
  - textures, samplers and bind groups
  - pipelines and passes
  - copies and external images
  - query sets, render bundles and labels
- **Zero-copy presentation on Linux** (dma-bufs shared with Skia).
- **CI.** A WPT import of a CTS slice that runs on lavapipe would be the natural next step. It is not done.
- **Robustness.** The root-cause fixes above come before any of this is fit to ship.
