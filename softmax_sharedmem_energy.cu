#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>
#include <thread>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_NVML(call) \
    do { \
        nvmlReturn_t err = call; \
        if (err != NVML_SUCCESS) { \
            fprintf(stderr, "NVML Error at %s:%d: %s\n", __FILE__, __LINE__, nvmlErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__
void softmax_forward_kernel(float *out, const float *inp, int N, int C)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;

    // ---- STEP 1: compute max ----
    float submax = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
        submax = fmaxf(submax, inp[row * C + i]);
    smem[tid] = submax;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    float max_val = smem[0];
    __syncthreads();

    // ---- STEP 2: compute exp and partial sum ----
    float subsum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float v = expf(inp[row * C + i] - max_val);
        out[row * C + i] = v;
        subsum += v;
    }
    smem[tid] = subsum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float sum_val = smem[0];
    __syncthreads();

    // ---- STEP 3: normalize ----
    for (int i = tid; i < C; i += blockDim.x)
        out[row * C + i] /= sum_val;
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

    size_t size = N * C * sizeof(float);

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
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, size));
    CHECK_CUDA(cudaMalloc(&d_y, size));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    int threads = 512;
    int shared = threads * sizeof(float);

    // -----------------------------------------------------
    // Initialize NVML
    // -----------------------------------------------------
    CHECK_NVML(nvmlInit());

    // Get device handle (assuming GPU 0)
    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));

    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    printf("GPU: %s\n", name);

    // -----------------------------------------------------
    // Warmup
    // -----------------------------------------------------
    const int WARMUP_ITERS = 100;
    for (int i = 0; i < WARMUP_ITERS; i++)
        softmax_forward_kernel<<<N, threads, shared>>>(d_y, d_x, N, C);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Warmup: %d iterations completed.\n", WARMUP_ITERS);

    // Wait a bit to let power stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // -----------------------------------------------------
    // Energy measurement for 100 rounds
    // -----------------------------------------------------
    const int RUN_ITERS = 1000;
    const int NUM_ROUNDS = 1000;

    std::vector<double> energy_measurements;
    energy_measurements.reserve(NUM_ROUNDS);

    printf("\n=== Running %d rounds of energy measurement ===\n", NUM_ROUNDS);
    printf("Each round runs %d kernel iterations\n", RUN_ITERS);

    for (int round = 0; round < NUM_ROUNDS; round++) {
        // Get initial power reading
        unsigned int power_start;
        CHECK_NVML(nvmlDeviceGetPowerUsage(device, &power_start));

        auto time_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < RUN_ITERS; i++) {
            softmax_forward_kernel<<<N, threads, shared>>>(d_y, d_x, N, C);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        auto time_end = std::chrono::high_resolution_clock::now();

        // Get final power reading
        unsigned int power_end;
        CHECK_NVML(nvmlDeviceGetPowerUsage(device, &power_end));

        // Calculate duration
        double duration_s = std::chrono::duration<double>(time_end - time_start).count();

        // Average power during execution (in milliwatts)
        double avg_power_mw = (power_start + power_end) / 2.0;
        double avg_power_w = avg_power_mw / 1000.0;

        // Energy = Power × Time
        double energy_j = avg_power_w * duration_s;
        energy_measurements.push_back(energy_j);

        if ((round + 1) % 10 == 0) {
            printf("Completed %d/%d rounds\n", round + 1, NUM_ROUNDS);
        }
    }

    // Calculate statistics
    double sum = 0.0;
    for (double e : energy_measurements) {
        sum += e;
    }
    double mean_energy = sum / NUM_ROUNDS;

    double variance = 0.0;
    for (double e : energy_measurements) {
        variance += (e - mean_energy) * (e - mean_energy);
    }
    double stddev_energy = std::sqrt(variance / NUM_ROUNDS);

    // Find min and max
    double min_energy = *std::min_element(energy_measurements.begin(), energy_measurements.end());
    double max_energy = *std::max_element(energy_measurements.begin(), energy_measurements.end());

    // Get power limit info
    unsigned int power_limit;
    nvmlDeviceGetPowerManagementLimit(device, &power_limit);

    printf("\n=== Energy Measurement Results ===\n");
    printf("Number of rounds: %d\n", NUM_ROUNDS);
    printf("Iterations per round: %d\n", RUN_ITERS);
    printf("GPU power limit: %.2f W\n\n", power_limit / 1000.0);

    printf("Energy per round (%d iterations):\n", RUN_ITERS);
    printf("  Mean:   %.6f J (±%.6f J)\n", mean_energy, stddev_energy);
    printf("  StdDev: %.6f J (%.2f%%)\n", stddev_energy, (stddev_energy / mean_energy) * 100.0);
    printf("  Min:    %.6f J\n", min_energy);
    printf("  Max:    %.6f J\n", max_energy);

    printf("\nEnergy per kernel iteration:\n");
    printf("  Mean:   %.9f J (%.6f mJ)\n", mean_energy / RUN_ITERS, (mean_energy / RUN_ITERS) * 1000.0);
    printf("  StdDev: %.9f J (%.6f mJ)\n", stddev_energy / RUN_ITERS, (stddev_energy / RUN_ITERS) * 1000.0);

    // -----------------------------------------------------
    // Validate results
    // -----------------------------------------------------
    CHECK_CUDA(cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost));
    softmax_cpu(h_x, h_ref, N, C);

    double max_err = 0.0;
    for (int i = 0; i < N*C; i++)
        max_err = std::max(max_err, (double)fabs(h_y[i] - h_ref[i]));
    if (max_err < 1e-4) printf("\nCorrectness checking passed!\n");
    else {
        printf("\nCorrectness checking failed!\n");
        return 1;
    }

    // cleanup
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    free(h_x);
    free(h_y);
    free(h_ref);

    CHECK_NVML(nvmlShutdown());

    return 0;
}
