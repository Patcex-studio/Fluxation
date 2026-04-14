#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sgd_step_kernel(float* params, const float* grads, float lr, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    params[idx] -= lr * grads[idx];
}

__global__ void compute_squared_norms_kernel(const float* params, float* losses, unsigned int num_params, unsigned int pop_size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) {
        return;
    }
    const float* candidate = params + (unsigned long long)idx * num_params;
    float sum = 0.0f;
    for (unsigned int j = 0; j < num_params; ++j) {
        float v = candidate[j];
        sum += v * v;
    }
    losses[idx] = sum;
}

extern "C" {
    int cuda_sgd_step_kernel_wrapper(float* params_ptr, const float* grad_ptr, float lr, unsigned int n) {
        if (!params_ptr || !grad_ptr || n == 0) {
            return -1;
        }

        float* d_params = nullptr;
        float* d_grads = nullptr;
        cudaError_t err = cudaMalloc((void**)&d_params, n * sizeof(float));
        if (err != cudaSuccess) {
            return -2;
        }
        err = cudaMalloc((void**)&d_grads, n * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_params);
            return -3;
        }

        err = cudaMemcpy(d_params, params_ptr, n * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_grads);
            return -4;
        }
        err = cudaMemcpy(d_grads, grad_ptr, n * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_grads);
            return -5;
        }

        unsigned int block = 256;
        unsigned int grid = (n + block - 1) / block;
        sgd_step_kernel<<<grid, block>>>(d_params, d_grads, lr, n);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_grads);
            return -6;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_grads);
            return -7;
        }

        err = cudaMemcpy(params_ptr, d_params, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_grads);
            return -8;
        }

        cudaFree(d_params);
        cudaFree(d_grads);
        return 0;
    }

    int cuda_compute_squared_norms_kernel_wrapper(
        const float* params_ptr,
        float* losses_ptr,
        unsigned int num_params,
        unsigned int pop_size,
    ) {
        if (!params_ptr || !losses_ptr || num_params == 0 || pop_size == 0) {
            return -1;
        }

        float* d_params = nullptr;
        float* d_losses = nullptr;
        unsigned int total_values = num_params * pop_size;
        cudaError_t err = cudaMalloc((void**)&d_params, total_values * sizeof(float));
        if (err != cudaSuccess) {
            return -2;
        }
        err = cudaMalloc((void**)&d_losses, pop_size * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_params);
            return -3;
        }

        err = cudaMemcpy(d_params, params_ptr, total_values * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_losses);
            return -4;
        }

        unsigned int block = 256;
        unsigned int grid = (pop_size + block - 1) / block;
        compute_squared_norms_kernel<<<grid, block>>>(d_params, d_losses, num_params, pop_size);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_losses);
            return -5;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_losses);
            return -6;
        }

        err = cudaMemcpy(losses_ptr, d_losses, pop_size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_params);
            cudaFree(d_losses);
            return -7;
        }

        cudaFree(d_params);
        cudaFree(d_losses);
        return 0;
    }
}
