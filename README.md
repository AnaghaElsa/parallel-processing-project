# Smart Parking System (NPP Accelerated)

This project implements a high-speed parking occupancy detection system using 
NVIDIA's NPP library and custom CUDA kernels for frame differencing.

## Performance
- **GPU Latency:** ~8ms
- **Throughput:** ~35 FPS (1080p)
- **Optimization:** NPP-accelerated color conversion and absolute difference.

## Requirements
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.x/12.x
- OpenCV (for video ingestion)
