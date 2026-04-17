#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

struct CudaOptimizerState {
    float* d_params;
    float* d_grads;
    unsigned int len;
    cudaStream_t stream;
    unsigned int block_size;
};

static const char* cuda_error_message(int error_code) {
    if (error_code == 0) {
        return nullptr;
    }

    if (error_code < 0) {
        cudaError_t cuda_code = (cudaError_t)(-error_code);
        const char* cuda_str = cudaGetErrorString(cuda_code);
        if (cuda_str != nullptr) {
            return cuda_str;
        }
    }

    switch (error_code) {
        case -1000:
            return "invalid CUDA optimizer state or invalid arguments";
        case -1001:
            return "failed to create CUDA stream";
        case -1002:
            return "failed to allocate persistent device parameters buffer";
        case -1003:
            return "failed to allocate temporary gradient buffer";
        case -1004:
            return "failed to copy data from host to device";
        case -1005:
            return "kernel launch failed";
        case -1006:
            return "CUDA stream synchronization failed";
        case -1007:
            return "failed to copy data from device to host";
        case -1008:
            return "failed to free CUDA resources";
        default:
            return "unknown CUDA wrapper error";
    }
}

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

static unsigned int choose_block_size(unsigned int requested, const void* kernel_func) {
    if (requested != 0) {
        return requested;
    }

    int minGridSize = 0;
    int blockSize = 0;
    cudaError_t occupancy_err = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        kernel_func,
        0,
        0
    );
    if (occupancy_err == cudaSuccess && blockSize > 0) {
        return (unsigned int)blockSize;
    }

    return 256u;
}

extern "C" {
    const char* cuda_error_message_wrapper(int error_code) {
        return cuda_error_message(error_code);
    }

    int cuda_init_cuda_optimizer_state(const float* params_ptr, unsigned int n, unsigned int block_size, struct CudaOptimizerState** out_state) {
        if (!out_state || !params_ptr || n == 0) {
            return -1000;
        }

        unsigned int chosen_block_size = choose_block_size(block_size, (const void*)sgd_step_kernel);
        cudaError_t err;

        struct CudaOptimizerState* state = (struct CudaOptimizerState*)malloc(sizeof(struct CudaOptimizerState));
        if (!state) {
            return -1000;
        }
        state->d_params = nullptr;
        state->d_grads = nullptr;
        state->len = n;
        state->block_size = chosen_block_size;
        state->stream = nullptr;

        err = cudaStreamCreate(&state->stream);
        if (err != cudaSuccess) {
            free(state);
            return -((int)err);
        }

        err = cudaMalloc((void**)&state->d_params, n * sizeof(float));
        if (err != cudaSuccess) {
            cudaStreamDestroy(state->stream);
            free(state);
            return -((int)err);
        }

        err = cudaMalloc((void**)&state->d_grads, n * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(state->d_params);
            cudaStreamDestroy(state->stream);
            free(state);
            return -((int)err);
        }

        err = cudaMemcpyAsync(state->d_params, params_ptr, n * sizeof(float), cudaMemcpyHostToDevice, state->stream);
        if (err != cudaSuccess) {
            cudaFree(state->d_params);
            cudaFree(state->d_grads);
            cudaStreamDestroy(state->stream);
            free(state);
            return -((int)err);
        }

        err = cudaStreamSynchronize(state->stream);
        if (err != cudaSuccess) {
            cudaFree(state->d_params);
            cudaFree(state->d_grads);
            cudaStreamDestroy(state->stream);
            free(state);
            return -((int)err);
        }

        *out_state = state;
        return 0;
    }

    int cuda_sgd_step_with_state_wrapper(
        struct CudaOptimizerState* state,
        float* params_ptr,
        const float* grad_ptr,
        float lr,
        int copy_back
    ) {
        if (!state || !state->d_params || !grad_ptr || state->len == 0) {
            return -1000;
        }

        cudaError_t err = cudaMemcpyAsync(state->d_grads, grad_ptr, state->len * sizeof(float), cudaMemcpyHostToDevice, state->stream);
        if (err != cudaSuccess) {
            return -((int)err);
        }

        unsigned int grid = (state->len + state->block_size - 1) / state->block_size;
        sgd_step_kernel<<<grid, state->block_size, 0, state->stream>>>(state->d_params, state->d_grads, lr, state->len);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_grads);
            return -((int)err);
        }

        if (copy_back && params_ptr) {
            err = cudaMemcpyAsync(params_ptr, state->d_params, state->len * sizeof(float), cudaMemcpyDeviceToHost, state->stream);
            if (err != cudaSuccess) {
                cudaFree(d_grads);
                return -((int)err);
            }
        }

        err = cudaStreamSynchronize(state->stream);
        cudaFree(d_grads);
        if (err != cudaSuccess) {
            return -((int)err);
        }

        return 0;
    }

    int cuda_free_optimizer_state(struct CudaOptimizerState* state) {
        if (!state) {
            return 0;
        }

        int result = 0;
        if (state->d_params) {
            cudaError_t err = cudaFree(state->d_params);
            if (err != cudaSuccess) {
                result = -((int)err);
            }
        }

        if (state->d_grads) {
            cudaError_t err = cudaFree(state->d_grads);
            if (err != cudaSuccess && result == 0) {
                result = -((int)err);
            }
        }

        if (state->stream) {
            cudaError_t err = cudaStreamDestroy(state->stream);
            if (err != cudaSuccess) {
                result = result == 0 ? -((int)err) : result;
            }
        }

        free(state);
        return result;
    }

    int cuda_sgd_step_kernel_wrapper(float* params_ptr, const float* grad_ptr, float lr, unsigned int n, unsigned int block_size) {
        struct CudaOptimizerState* state = nullptr;
        int init_result = cuda_init_cuda_optimizer_state(params_ptr, n, block_size, &state);
        if (init_result != 0) {
            return init_result;
        }

        int result = cuda_sgd_step_with_state_wrapper(state, params_ptr, grad_ptr, lr, 1);
        int free_result = cuda_free_optimizer_state(state);
        if (result != 0) {
            return result;
        }
        return free_result;
    }

    int cuda_compute_squared_norms_kernel_wrapper(
        const float* params_ptr,
        float* losses_ptr,
        unsigned int num_params,
        unsigned int pop_size,
        unsigned int block_size
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

        unsigned int block = choose_block_size(block_size, (const void*)compute_squared_norms_kernel);
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
