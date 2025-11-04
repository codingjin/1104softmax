#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// warp-level reduction helpers (works for warp size=32)
__inline__ __device__
float warp_reduce_max(float val) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    return __shfl_sync(mask, val, 0);
}

__inline__ __device__
float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return __shfl_sync(mask, val, 0);
}

__global__
void softmax_forward_kernel_float4(float *out, const float *inp, int N, int C)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = (blockDim.x + 31) / 32;

    int vecC = C / 4; // number of float4 elements per row
    const float4* inp4 = reinterpret_cast<const float4*>(inp + row * C);
    float4* out4 = reinterpret_cast<float4*>(out + row * C);

    // STEP 1: compute local max
    float local_max = -INFINITY;
    for (int i = tid; i < vecC; i += blockDim.x) {
        float4 v = inp4[i];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }

    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) smem[warpId] = warp_max;
    __syncthreads();

    float max_val;
    if (warpId == 0) {
        float val = (lane < numWarps) ? smem[lane] : -INFINITY;
        float w0 = warp_reduce_max(val);
        if (lane == 0) smem[0] = w0;
    }
    __syncthreads();
    max_val = smem[0];

    // STEP 2: compute exp(x - max) and partial sum
    float local_sum = 0.0f;
    for (int i = tid; i < vecC; i += blockDim.x) {
        float4 v = inp4[i];
        float e0 = expf(v.x - max_val);
        float e1 = expf(v.y - max_val);
        float e2 = expf(v.z - max_val);
        float e3 = expf(v.w - max_val);
        out4[i] = make_float4(e0, e1, e2, e3);
        local_sum += e0 + e1 + e2 + e3;
    }

    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smem[warpId] = warp_sum;
    __syncthreads();

    float sum_val;
    if (warpId == 0) {
        float val = (lane < numWarps) ? smem[lane] : 0.0f;
        float w0 = warp_reduce_sum(val);
        if (lane == 0) smem[0] = w0;
    }
    __syncthreads();
    sum_val = smem[0];

    // STEP 3: normalize
    float inv_sum = 1.0f / sum_val;
    for (int i = tid; i < vecC; i += blockDim.x) {
        float4 e4 = out4[i];
        e4.x *= inv_sum; e4.y *= inv_sum; e4.z *= inv_sum; e4.w *= inv_sum;
        out4[i] = e4;
    }
}

// CPU reference (numerically stable)
void softmax_cpu(const float* x, float* y, int N, int C) {
    for (int n = 0; n < N; n++) {
        float maxv = -INFINITY;
        for (int j = 0; j < C; j++) maxv = std::max(maxv, x[n*C + j]);
        float sum = 0.0f;
        for (int j = 0; j < C; j++) sum += std::exp(x[n*C + j] - maxv);
        for (int j = 0; j < C; j++) y[n*C + j] = std::exp(x[n*C + j] - maxv) / sum;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) { std::cout << "Usage: ./softmax N C\n"; return 1; }
    int N = std::atoi(argv[1]);
    int C = std::atoi(argv[2]);
    if (C % 4 != 0) { std::cerr << "C must be divisible by 4 for float4 kernel.\n"; return 1; }

    size_t size = (size_t)N * C * sizeof(float);
    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);

    for (int i = 0; i < N*C; i++)
        h_x[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;

    float *d_x = nullptr, *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, size));
    CHECK_CUDA(cudaMalloc(&d_y, size));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    int threads = 512;//256;
    int numWarps = (threads + 31) / 32;
    int shared = numWarps * sizeof(float);

    // -----------------------------------------------------
    // Warmup
    // -----------------------------------------------------
    const int WARMUP_ITERS = 100;
    for (int i = 0; i < WARMUP_ITERS; i++)
        softmax_forward_kernel_float4<<<N, threads, shared>>>(d_y, d_x, N, C);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Warmup: %d iterations completed.\n", WARMUP_ITERS);

    // -----------------------------------------------------
    // Timed runs
    // -----------------------------------------------------
    const int RUN_ITERS = 1000;
    std::vector<float> times(RUN_ITERS);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < RUN_ITERS; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        softmax_forward_kernel_float4<<<N, threads, shared>>>(d_y, d_x, N, C);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&times[i], start, stop));
    }

    // compute mean, stddev, ratio
    double sum_t = 0.0;
    for (auto t : times) sum_t += t;
    double mean = sum_t / RUN_ITERS;

    double var = 0.0;
    for (auto t : times) var += (t - mean)*(t - mean);
    double stddev = std::sqrt(var / RUN_ITERS);
    double ratio = (stddev / mean) * 100.0;
    printf("Average GPU kernel time: %.6f ms ± %.6f ms (%.2f%%) over %d runs\n",
           mean, stddev, ratio, RUN_ITERS);

    // -----------------------------------------------------
    // GFLOPs estimation
    // -----------------------------------------------------
    double flops_per_instance = 5.0 * (double)N * (double)C;
    double gflops = (flops_per_instance / (mean / 1000.0)) / 1e9;
    printf("Approx. GFLOPs: %.3f\n", gflops);

    // -----------------------------------------------------
    // Validate results
    // -----------------------------------------------------
    CHECK_CUDA(cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost));
    softmax_cpu(h_x, h_ref, N, C);

    double max_err = 0.0;
    for (int i = 0; i < N*C; i++) max_err = std::max(max_err, (double)fabs(h_y[i]-h_ref[i]));
    //printf("Max error = %e\n", max_err);
    if (max_err < 1e-4) printf("Correctness checking passed!\n");
    else {
        printf("Correctness checking failed!\n");
        return 1;
    }

    // cleanup
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    free(h_x); free(h_y); free(h_ref);
    return 0;
}
