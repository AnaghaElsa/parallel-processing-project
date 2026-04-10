/**
 * @file parking_npp.cu
 * @brief GPU-accelerated parking occupancy detection using NPP and Custom Kernels.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_threshold_and_compare_operations.h>

#define WIDTH    1920
#define HEIGHT   1080
#define N_SLOTS  30
#define CHANNELS 3

/**
 * Custom CUDA kernel for parallel ROI occupancy detection.
 * Uses shared memory reduction to calculate standard deviation within a specific ROI.
 */
__global__ void computeSlotOccupancy(
    const Npp8u* d_gray,
    float* d_std_out,
    const int* d_sx, const int* d_sy,
    const int* d_sw, const int* d_sh,
    int img_width)
{
    int s   = blockIdx.x; 
    int tid = threadIdx.x;

    // ROI insets to reduce noise from parking line markers
    int x     = d_sx[s] + 6;
    int y     = d_sy[s] + 15;
    int w     = d_sw[s] - 12;
    int h     = d_sh[s] - 30;
    int total = w * h;

    __shared__ float s_sum[256];
    __shared__ float s_sq [256];

    float lsum = 0.0f, lsq = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        float v = (float)d_gray[(y + i/w) * img_width + (x + i%w)];
        lsum += v;
        lsq  += v * v;
    }
    s_sum[tid] = lsum;
    s_sq [tid] = lsq;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq [tid] += s_sq [tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / total;
        float var  = (s_sq[0] / total) - (mean * mean);
        d_std_out[s] = sqrtf(var < 0 ? 0 : var);
    }
}

/**
 * Kernel to convert BGR (OpenCV default) to RGB for NPP compatibility.
 */
__global__ void bgrToRgbKernel(const Npp8u* bgr, Npp8u* rgb, int n_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    rgb[i*3+0] = bgr[i*3+2];
    rgb[i*3+1] = bgr[i*3+1];
    rgb[i*3+2] = bgr[i*3+0];
}

int main() {
    // ROI coordinates for 30 slots (Upper and Lower rows)
    int h_sx[N_SLOTS], h_sy[N_SLOTS], h_sw[N_SLOTS], h_sh[N_SLOTS];
    for (int i = 0; i < 15; i++) {
        int x = 50 + i * 97;
        h_sx[i]    = x+4; h_sy[i]    = 325; h_sw[i]    = 85; h_sh[i]    = 120;
        h_sx[i+15] = x+4; h_sy[i+15] = 648; h_sw[i+15] = 85; h_sh[i+15] = 175;
    }

    size_t bgr_size  = WIDTH * HEIGHT * CHANNELS;
    size_t gray_size = WIDTH * HEIGHT;
    Npp8u *h_frame1 = (Npp8u*)malloc(bgr_size);
    Npp8u *h_frame2 = (Npp8u*)malloc(bgr_size);

    // Ingest frames from binary storage
    FILE* f1 = fopen("frame1.bin", "rb");
    FILE* f2 = fopen("frame2.bin", "rb");
    if (!f1 || !f2) return -1;
    fread(h_frame1, 1, bgr_size, f1);
    fread(h_frame2, 1, bgr_size, f2);
    fclose(f1); fclose(f2);

    // Device memory management
    Npp8u *d_bgr, *d_rgb, *d_gray1, *d_gray2, *d_diff, *d_thresh;
    float *d_std;
    int *d_sx_g, *d_sy_g, *d_sw_g, *d_sh_g;

    cudaMalloc(&d_bgr, bgr_size);
    cudaMalloc(&d_rgb, bgr_size);
    cudaMalloc(&d_gray1, gray_size);
    cudaMalloc(&d_gray2, gray_size);
    cudaMalloc(&d_diff, gray_size);
    cudaMalloc(&d_thresh, gray_size);
    cudaMalloc(&d_std, N_SLOTS * sizeof(float));
    cudaMalloc(&d_sx_g, N_SLOTS * sizeof(int));
    cudaMalloc(&d_sy_g, N_SLOTS * sizeof(int));
    cudaMalloc(&d_sw_g, N_SLOTS * sizeof(int));
    cudaMalloc(&d_sh_g, N_SLOTS * sizeof(int));

    cudaMemcpy(d_sx_g, h_sx, N_SLOTS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sy_g, h_sy, N_SLOTS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sw_g, h_sw, N_SLOTS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh_g, h_sh, N_SLOTS*sizeof(int), cudaMemcpyHostToDevice);

    NppiSize roiSize = {WIDTH, HEIGHT};
    int npix = WIDTH * HEIGHT;

    // Process Background Reference (Frame 1)
    cudaMemcpy(d_bgr, h_frame1, bgr_size, cudaMemcpyHostToDevice);
    bgrToRgbKernel<<<(npix+255)/256, 256>>>(d_bgr, d_rgb, npix);
    nppiRGBToGray_8u_C3C1R(d_rgb, WIDTH*3, d_gray1, WIDTH, roiSize);

    // Process Current Frame (Frame 2)
    cudaMemcpy(d_bgr, h_frame2, bgr_size, cudaMemcpyHostToDevice);
    bgrToRgbKernel<<<(npix+255)/256, 256>>>(d_bgr, d_rgb, npix);
    nppiRGBToGray_8u_C3C1R(d_rgb, WIDTH*3, d_gray2, WIDTH, roiSize);

    // Frame Differencing and Thresholding
    nppiAbsDiff_8u_C1R(d_gray2, WIDTH, d_gray1, WIDTH, d_diff, WIDTH, roiSize);
    nppiThreshold_Val_8u_C1R(d_diff, WIDTH, d_thresh, WIDTH, roiSize, 25, 255, NPP_CMP_GREATER);

    // Occupancy Analysis
    computeSlotOccupancy<<<N_SLOTS, 256>>>(d_gray2, d_std, d_sx_g, d_sy_g, d_sw_g, d_sh_g, WIDTH);
    
    float h_std[N_SLOTS];
    cudaMemcpy(h_std, d_std, N_SLOTS*sizeof(float), cudaMemcpyDeviceToHost);

    // Final Reporting
    int occupied = 0;
    printf("Slot Occupancy Report:\n----------------------\n");
    for (int i=0; i<N_SLOTS; i++) {
        float thr = (i < 15) ? 32.0f : 27.0f;
        bool isOccupied = h_std[i] > thr;
        if (isOccupied) occupied++;
        printf("Slot %d: %s (StdDev: %.1f)\n", i+1, isOccupied ? "OCCUPIED" : "EMPTY", h_std[i]);
    }

    printf("\nTotal Occupied: %d / %d\n", occupied, N_SLOTS);

    // Cleanup
    cudaFree(d_bgr); cudaFree(d_rgb); cudaFree(d_gray1); cudaFree(d_gray2);
    cudaFree(d_diff); cudaFree(d_thresh); cudaFree(d_std);
    cudaFree(d_sx_g); cudaFree(d_sy_g); cudaFree(d_sw_g); cudaFree(d_sh_g);
    free(h_frame1); free(h_frame2);

    return 0;
}
