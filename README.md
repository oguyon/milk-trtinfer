# trtinfer: TensorRT & Neural Network Inference Plugin for milk

An optional, high-performance plugin for [milk](https://github.com/milk-org/milk) providing generic real-time neural network inference powered by NVIDIA TensorRT (GPU) and native BLAS / SIMD (CPU), operating directly on `ImageStreamIO` shared memory streams.

> [!WARNING]
> **Status & Experimental Disclaimer**
> 1. **Proof of Concept**: This plugin is an experimental proof of concept exploring real-time neural network inference directly on `ImageStreamIO` shared memory streams.
> 2. **`framework-dev` Standards in Flux**: It leverages `milk`'s new `framework-dev` architecture and FPS V2 interface conventions, which are actively evolving and still subject to changes.
> 3. **Agentic Implementation**: This repository was implemented using AI agentic coding tools (directed and reviewed by O. Guyon).
> 4. **Testing Status**: While self-contained verification suites are included, this codebase has **not** yet been thoroughly tested across diverse production environments, GPU architectures, or edge cases.
> 
> **Feedback & Contributions Welcome**: Comments, architectural suggestions, performance feedback, and bug reports are warmly encouraged to help refine and harden this plugin.

---

## 1. Overview & Architecture

`trtinfer` bridges deep learning inference and real-time astronomical adaptive optics (AO) / wavefront control. By operating directly on `ImageStreamIO` shared memory arrays (`/milk/shm` / `/dev/shm`), it enables sub-millisecond, low-jitter neural network predictions synchronized to telemetry streams.

### Key Capabilities
- **Zero Python Runtime**: All runtime inference, data streaming, and benchmark training pipelines are implemented in 100% compiled C/C++.
- **Direct ImageStreamIO Integration**: Connects seamlessly with standard POSIX shared memory streams (`/dev/shm/*.im.shm`). Fully compatible with `milk-streamCTRL`, `shmimview`, and `listim`.
- **FPS V2 Architecture**: Built with milk's unified Function Parameter Structure (FPS V2), providing both interactive milk CLI commands and standalone executables (`milk-fpsexec-*`).
- **Flexible Model Support**:
  - **Standalone Compact Models (`.bin`)**: Self-contained weights trained directly in C from shared memory with zero external dependencies.
  - **TensorRT Engines (`.engine`)**: Optimized CUDA inference plans serialized via TensorRT.
  - **ONNX Models (`.onnx`)**: Standard ONNX models including CNNs, U-Nets, ResNets, and MLPs parsed via `nvonnxparser`.

---

## 2. Components & Executables

| Component | Target Name | Description |
| :--- | :--- | :--- |
| **Inference Compute Unit** | `milk-fpsexec-trtinfer` | FPS V2 real-time inference engine on ImageStreamIO |
| **Dataset Generator** | `milk-fpsexec-trtinfer_datagen` | Generates non-linear 3D optical swirl datasets & trajectories |
| **C Model Trainer** | `trtinfer_train_mlp` | Standalone C trainer (Adam optimizer) directly on shared memory |
| **C Accuracy Validator** | `trtinfer_check_accuracy` | Verifies data and inference accuracy against analytical formulas |
| **Shared Library** | `libmilktrtinfer.so` | Milk plugin shared library providing CLI commands |

---

## 3. User Guide & Testing

### 3.1 Building the Plugin

From the root of your `milk` build tree:
```bash
cmake -B _build
make -C _build -j4 milktrtinfer milk-fpsexec-trtinfer milk-fpsexec-trtinfer_datagen trtinfer_train_mlp trtinfer_check_accuracy
```

---

### 3.2 Option 1: One-Command Automated Test (Recommended)

Run the self-contained end-to-end verification script from the milk root directory:
```bash
./plugins/trtinfer/scripts/test_end_to_end.sh
```

In about **7 seconds**, this script automatically:
1. Verifies `-h1` one-line help compliance for all standalone binaries.
2. Generates 5,000 non-linear 3D optical swirl points in `ImageStreamIO` shared memory (`/milk/shm/` or `/dev/shm/`).
3. Mathematically validates the data points against analytical equations ($0.000\times 10^0$ error).
4. Trains a 3-layer MLP neural network ($3 \to 64 \to 64 \to 3$) directly from `ImageStreamIO` shared memory using mini-batch Adam in pure C ($\approx 3.9$ s).
5. Runs real-time inference via `milk-fpsexec-trtinfer` on the input stream to create the output stream.
6. Verifies that the neural network predictions match the ground-truth stream to high accuracy ($\text{RMSE} \approx 0.007$, well below the $0.05$ threshold).
7. Cleans up all temporary test streams and files.

---

### 3.3 Option 2: Step-by-Step Interactive Example

Run each stage manually in your terminal to inspect the shared memory streams in real time:

#### Step 1: Generate 5,000 3D samples in ImageStreamIO shared memory
```bash
./_build/plugins/trtinfer/milk-fpsexec-trtinfer_datagen -n datagen exec pts_in pts_truth 5000 /tmp/data_3d
```
*Creates two shared memory streams: `pts_in` (inputs $[3 \times 5000]$) and `pts_truth` (ground truth $[3 \times 5000]$).*

#### Step 2: Inspect active shared memory streams
```bash
./_build/_install/bin/milk-stream-list
```
*(You will see `pts_in` and `pts_truth` listed as `3 x 5000` float streams).*

#### Step 3: Train the neural network directly from shared memory in pure C
```bash
./_build/plugins/trtinfer/trtinfer_train_mlp -i pts_in -t pts_truth -o /tmp/model_3d.bin -e 500 -b 64 -r 0.005
```
*Connects directly to `pts_in` and `pts_truth`, trains the MLP with Adam in $\approx 3.9$ seconds, and writes the compact trained model to `/tmp/model_3d.bin` (18.5 KB).*

#### Step 4: Run real-time inference to produce `pts_out`
```bash
./_build/plugins/trtinfer/milk-fpsexec-trtinfer -n infer exec pts_in pts_out /tmp/model_3d.bin
```
*`milk-fpsexec-trtinfer` connects to `pts_in`, evaluates the neural network, creates output stream `pts_out` $[3 \times 5000]$, and posts semaphores.*

#### Step 5: Verify the numerical accuracy of the predictions
```bash
./_build/plugins/trtinfer/trtinfer_check_accuracy pts_in pts_truth none pts_out
```
*Expected output:*
```text
Dataset validation:
  Samples verified:                    5000
  Max discrepancy vs analytical model: 0.000e+00
  RMSE vs analytical model:            0.000e+00
Inference accuracy validation ('pts_out' vs 'pts_truth'):
  Inference Max Absolute Error:        0.0313
  Inference RMSE:                      0.0067 (Pass threshold: < 0.05)
  Inference Accuracy Check:            [PASS]
```

---

### 3.4 Option 3: Continuous 100 Hz Streaming Mode

To test continuous trajectory streaming with real-time semaphore synchronization:

**Terminal 1 (Stream Generator at 100 Hz):**
```bash
./_build/plugins/trtinfer/milk-fpsexec-trtinfer_datagen -n datagen set pts_in pts_truth 0 2 "" 100.0
./_build/plugins/trtinfer/milk-fpsexec-trtinfer_datagen -n datagen exec
```

**Terminal 2 (Real-Time Inference triggered on stream semaphores):**
```bash
./_build/plugins/trtinfer/milk-fpsexec-trtinfer -n infer exec pts_in pts_out /tmp/model_3d.bin
```

**Terminal 3 (Monitor latency and semaphores):**
```bash
./_build/_install/bin/milk-streamCTRL
```

---

## 4. 3D Optical Swirl Benchmark

To provide an exact, reproducible ground truth for non-linear regression benchmarks without external data files, `trtinfer_datagen` implements a coupled non-linear coordinate transformation:

$$\begin{aligned}
r_{12} &= \sqrt{x_1^2 + x_2^2 + 10^{-6}} \\
\theta &= \frac{\pi}{2} \cdot r_{12} \\
y_1 &= x_1 \cos(\theta) - x_2 \sin(\theta) + 0.25 \, x_3^2 \\
y_2 &= x_1 \sin(\theta) + x_2 \cos(\theta) - 0.25 \, x_3^2 \\
y_3 &= \tanh(1.2 \, x_3) + 0.35 \sin(\pi \, x_1 x_2)
\end{aligned}$$

- **Input Domain**: $x \in [-1, 1]^3$
- **Output Range**: $y \in [-1.5, 1.5]^3$
- **Properties**: Strong non-linear coordinate coupling between $(x_1, x_2)$ rotation and $x_3$ quadratic warping, challenging for linear reconstructors and ideal for verifying neural networks.

---

## 5. Exporting Complex External Models (PyTorch $\to$ ONNX)

`trtinfer` supports complex external architectures (CNNs, U-Nets, ResNets) exported from PyTorch via standard ONNX.

### Example: Exporting a Wavefront Reconstruction U-Net
```python
import torch
import torch.nn as nn

class WavefrontUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-channel WFS input (e.g. 64x64) -> DM command map (e.g. 64x64)
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.out  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        d1 = self.relu(self.dec1(e2))
        return self.out(torch.cat([d1, e1], dim=1))

model = WavefrontUNet().eval()
dummy_input = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    model,
    dummy_input,
    "wavefront_unet.onnx",
    input_names=["wfs_in"],
    output_names=["dm_out"],
    opset_version=17
)
print("Exported wavefront_unet.onnx successfully.")
```

### Running the ONNX Model in `trtinfer`
Once exported, run directly with `milk-fpsexec-trtinfer`:
```bash
./_build/plugins/trtinfer/milk-fpsexec-trtinfer -n unet_recon exec wfs_stream dm_cmd_stream wavefront_unet.onnx
```
TensorRT compiles and optimizes the network into an execution engine on the target GPU, automatically streaming frames from `wfs_stream` to `dm_cmd_stream`.

---

## 6. Real-Time Streaming & Synchronization

`milk-fpsexec-trtinfer` supports two execution synchronization modes via the `.sync_mode` parameter:

1. **Trigger Mode (`sync_mode = 0`)**:
   - Synchronizes directly on the input stream's semaphores.
   - Computes inference as soon as a new frame is posted to `in_stream`.
   - Posts semaphores on `out_stream` immediately upon completion.
   - Ideal for closed-loop AO systems.

2. **Free-Run Timer Mode (`sync_mode = 1`)**:
   - Executes at a fixed user-configured rate (`.rate_hz`, e.g. 1000.0 Hz).
   - Samples the latest available frame from `in_stream`.

### Monitoring Latency
Check execution latency and sample counts in real-time via FPS CLI or `milk-streamCTRL`:
```bash
./_build/_install/bin/milk-fps-get trtinfer .latency_us
./_build/_install/bin/milk-fps-get trtinfer .sample_count
```

---

## 7. Standalone Plugin Repository

`trtinfer` is maintained as an independent git repository inside `plugins/trtinfer/` and is untracked by the parent `milk` repository.

```bash
cd plugins/trtinfer
git status
git log -n 5
```
All build files and headers are automatically detected by milk's root CMake build system.
