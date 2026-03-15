# Performance Benchmarks: Edge ML Optimization

## 1. Latency Reduction
By converting models to TensorRT engines, we typically observe a significant decrease in inference latency.
- **PyTorch (Baseline):** 120ms
- **TensorRT FP16:** 28ms (~4.2x speedup)
- **TensorRT INT8:** 14ms (~8.5x speedup)

## 2. Throughput Optimization (FPS)
Optimized engines allow for higher frame rates on NVIDIA Jetson modules (Orin/Xavier).
- **Orin Nano (FP16):** 45 FPS
- **Orin AGX (INT8):** 120+ FPS

## 3. Precision Analysis
KLD (Kullback�Leibler divergence) is used to measure the information loss during INT8 quantization. Our pipeline targets a precision drop of < 1% for mission-critical vision models.
