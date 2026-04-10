Smart Parking System: CUDA + NPP Accelerated Pipeline

A high-performance computer vision pipeline for real-time parking occupancy detection. This system leverages NVIDIA's NPP (Performance Primitives) and custom CUDA kernels to achieve a 3.5x speedup over traditional CPU-based processing.

## Performance Highlights
- Latency: ~8ms per frame
- Throughput: ~35 FPS (1080p Resolution)
- Architecture: Hybrid Python/C++ CUDA pipeline
- Hardware: Tested on NVIDIA T4 (Google Colab) / RTX Series

## Requirements
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.x/12.x
- OpenCV (for video ingestion)
