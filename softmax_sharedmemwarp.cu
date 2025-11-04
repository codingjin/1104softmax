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

// warp-level reduction helpers (works for power-of-two warp size = 32)
__inline__ __device__
float warp_reduce_max(float val) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, val, offset);
        val = fmaxf(val, other);
    }
    // after loop, lane 0 contains the warp max; broadcast lane0 to all lanes if desired
    return __shfl_sync(mask, val, 0);
}

__inline__ __device__
float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, val, offset);
        val = val + other;
    }
    return __shfl_sync(mask, val, 0);
}

__global__
void softmax_forward_kernel(float *out, const float *inp, int N, int C)
{
    // shared memory holds one float per warp for the inter-warp reduction
    extern __shared__ float smem[]; // size = numWarps * sizeof(float)

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int lane = tid & 31;                  // lane index in warp
    int warpId = tid >> 5;                // warp index in block
    int numWarps = (blockDim.x + 31) / 32;

    // -------------------------
    // STEP 1: per-thread local max over stride
    // -------------------------
    float local_max = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        local_max = fmaxf(local_max, inp[row * C + i]);
    }

    // warp-level max
    float warp_max = warp_reduce_max(local_max);

    // lane 0 of each warp writes warp_max to shared memory
    if (lane == 0) {
        smem[warpId] = warp_max;
    }
    __syncthreads();

    // inter-warp reduction: let first warp reduce the warp maxima
    float max_val;
    if (warpId == 0) {
        float val = -INFINITY;
        // each thread in first warp loads one element (if exists)
        int idx = lane;
        if (idx < numWarps) val = smem[idx];
        else val = -INFINITY;
        // warp-level reduce in first warp
        float warp0_max = warp_reduce_max(val); // lane 0 now has final max
        // broadcast by writing to smem[0] by lane 0
        if (lane == 0) smem[0] = warp0_max;
    }
    __syncthreads();

    max_val = smem[0]; // all threads read final max

    // -------------------------
    // STEP 2: compute exp(x - max) and partial sum
    // -------------------------
    float local_sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float v = expf(inp[row * C + i] - max_val);
        out[row * C + i] = v; // store numerator temporarily
        local_sum += v;
    }

    // warp-level sum
    float warp_sum = warp_reduce_sum(local_sum);

    // lane 0 writes warp_sum to shared memory
    if (lane == 0) {
        smem[warpId] = warp_sum;
    }
    __syncthreads();

    // inter-warp reduction for sum (first warp reduces warp sums)
    float sum_val;
    if (warpId == 0) {
        float val = 0.0f;
        int idx = lane;
        if (idx < numWarps) val = smem[idx];
        else val = 0.0f;
        float warp0_sum = warp_reduce_sum(val); // lane 0 holds the total sum
        if (lane == 0) smem[0] = warp0_sum;
    }
    __syncthreads();

    sum_val = smem[0];

    // -------------------------
    // STEP 3: normalize
    // -------------------------
    for (int i = tid; i < C; i += blockDim.x) {
        out[row * C + i] /= sum_val;
    }
}

// CPU reference (numerically stable)
void softmax_cpu(const float* x, float* y, int N, int C) {
    for (int n = 0; n < N; n++) {
        float maxv = -INFINITY;
        for (int j = 0; j < C; j++)
            maxv = std::max(maxv, x[n*C + j]);

        float sum = 0.0f;
        for (int j = 0; j < C; j++)
            sum += std::exp(x[n*C + j] - maxv);

        for (int j = 0; j < C; j++)
            y[n*C + j] = std::exp(x[n*C + j] - maxv) / sum;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "Usage: ./softmax N C\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    int C = std::atoi(argv[2]);
    printf("Softmax test: N=%d, C=%d\n", N, C);

    size_t size = (size_t)N * C * sizeof(float);

    // host memory
    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);
    if (!h_x || !h_y || !h_ref) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    // initialize input
    for (int i = 0; i < N*C; i++)
        h_x[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;

    // device memory
    float *d_x = nullptr, *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, size));
    CHECK_CUDA(cudaMalloc(&d_y, size));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    int threads = 512;//256;
    int numWarps = (threads + 31) / 32;
    int shared = numWarps * sizeof(float); // one float per warp for inter-warp reduction

    // -----------------------------------------------------
    // Warmup
    // -----------------------------------------------------
    const int WARMUP_ITERS = 100;
    for (int i = 0; i < WARMUP_ITERS; i++)
        softmax_forward_kernel<<<N, threads, shared>>>(d_y, d_x, N, C);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Warmup: %d iterations completed.\n", WARMUP_ITERS);

    // -----------------------------------------------------
    // Timed runs
    // -----------------------------------------------------
    const int RUN_ITERS = 1000;
    std::vector<float> times;
    times.reserve(RUN_ITERS);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < RUN_ITERS; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        softmax_forward_kernel<<<N, threads, shared>>>(d_y, d_x, N, C);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    // compute mean, stddev, and ratio
    double sum_t = 0.0;
    for (auto t : times) sum_t += t;
    double mean = sum_t / RUN_ITERS;

    double var = 0.0;
    for (auto t : times) var += (t - mean) * (t - mean);
    double stddev = std::sqrt(var / RUN_ITERS);
    double ratio = (stddev / mean) * 100.0; // percentage

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
    for (int i = 0; i < N*C; i++)
        max_err = std::max(max_err, (double)fabs(h_y[i] - h_ref[i]));
    //printf("Max error = %e\n", max_err);
    if (max_err < 1e-4) printf("Correctness checking passed!\n");
    else {
        printf("Correctness checking failed!\n");
        return 1;
    }

    // cleanup
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    free(h_x);
    free(h_y);
    free(h_ref);

    return 0;
}
