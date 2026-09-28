# trtinfer: TensorRT & Neural Network Inference Plugin for milk

An optional, high-performance plugin for [milk](https://github.com/milk-org/milk) providing generic real-time neural network inference powered by NVIDIA TensorRT (GPU) and ONNX Runtime / native BLAS (CPU), operating on native `ImageStreamIO` shared memory streams.

## Features

- **Generic Model Execution**: Run arbitrary neural network models directly on milk shared memory streams.
- **ImageStreamIO Integration**: Connects seamlessly with standard POSIX shared memory streams (`/dev/shm/*.im.shm`). Full compatibility with `milk-streamCTRL`, `shmimview`, and `listim`.
- **FPS V2 Architecture**: Built with milk's unified FPS-CLI V2 architecture, providing both interactive milk CLI commands and standalone executables (`milk-fpsexec-*`).
- **Benchmark Dataset Generator**: Includes `trtinfer_datagen` for generating non-linear 3D coordinate transformation datasets in batch or streaming modes.

---

## 3D Benchmark Dataset Generator (`trtinfer_datagen`)

To benchmark neural network models and real-time inference latency, `trtinfer` includes a coupled non-linear 3D $\to$ 3D coordinate mapping based on optical swirl warping:

$$\begin{aligned}
r_{12} &= \sqrt{x_1^2 + x_2^2 + 10^{-6}} \\
\theta &= \frac{\pi}{2} \cdot r_{12} \\
y_1 &= x_1 \cos(\theta) - x_2 \sin(\theta) + 0.25 \, x_3^2 \\
y_2 &= x_1 \sin(\theta) + x_2 \cos(\theta) - 0.25 \, x_3^2 \\
y_3 &= \tanh(1.2 \, x_3) + 0.35 \sin(\pi \, x_1 x_2)
\end{aligned}$$

### Usage

#### 1. Standalone Executable
```bash
# Generate 10,000 samples into ImageStreamIO streams 'pts_in' and 'pts_truth'
milk-fpsexec-trtinfer_datagen exec

# Customize parameters
milk-fpsexec-trtinfer_datagen set pts_in pts_truth 50000 0 "mydata.fits" 0.0
milk-fpsexec-trtinfer_datagen exec

# Continuous 1 kHz streaming trajectory mode
milk-fpsexec-trtinfer_datagen set pts_in pts_truth 0 2 "" 1000.0
milk-fpsexec-trtinfer_datagen exec
```

#### 2. Milk CLI
```
milk> trtinfer_datagen pts_in pts_truth 10000 0 "data_3d.fits" 0.0
```

#### 3. Monitoring
Monitor the active streams in real-time:
```bash
milk-streamCTRL
```

---

## Repository & Development

This plugin is designed to be maintained as an independent repository:
```bash
cd plugins/trtinfer
git status
```
Parent `milk` builds automatically discover and build this plugin via CMake auto-discovery.
