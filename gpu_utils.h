#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include "../wav_header.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

enum class VecMode {
    Scalar = 1, // Standard int16_t access (No alignment requirement)
    Int2   = 4, // Loading 4 shorts (8 bytes) via int2 type
    Int4   = 8  // Loading 8 shorts (16 bytes) via int4 type
};

class AveragerWorkspace {
private:
    int16_t* d_input_base = nullptr;  // The actual pointer to free
    int16_t* d_output_base = nullptr; // The actual pointer to free

    static size_t alignUp(size_t x, size_t alignment) {
        return (x + alignment - 1) & ~(alignment - 1);
    }

    // Disable Copying (Crucial! We don't want two objects freeing the same pointer)
    AveragerWorkspace(const AveragerWorkspace&) = delete;
    AveragerWorkspace& operator=(const AveragerWorkspace&) = delete;

public:
    // Public Accessors (Read-Only)
    int16_t* input_valid = nullptr; // Pointer to valid data start (after padding)
    int16_t* output = nullptr;      // Pointer to output
    const size_t halo_elements;     // The padded/aligned halo size
    const size_t total_alloc_elements; // Total buffer size (including end padding)
    const size_t valid_bytes;

    // 1. CONSTRUCTOR (Replaces allocate_gpu_resources)
    AveragerWorkspace(size_t num_samples, int grade, int num_channels, VecMode mode = VecMode::Scalar)
         : halo_elements(calculateHalo(grade, num_channels, mode)),
          total_alloc_elements(calculateTotal(num_samples, halo_elements, mode)),
          valid_bytes(num_samples * sizeof(int16_t)) {

        size_t total_input_bytes = total_alloc_elements * sizeof(int16_t);

        // Allocation
        CUDA_CHECK(cudaMalloc(&d_input_base, total_input_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_base, valid_bytes));

        // Initialization (Zero the apron)
        // 1. Zero the Halo (The Start)
        // Size: halo_elements
        if (halo_elements > 0) {
            CUDA_CHECK(cudaMemset(d_input_base, 0, halo_elements * sizeof(int16_t)));
        }

        // 2. Zero the Tail (The Vectorization Padding)
        // We only need to do this if total_alloc > (halo + samples)
        size_t used_elements = halo_elements + num_samples;
        if (total_alloc_elements > used_elements) {
            size_t tail_elements = total_alloc_elements - used_elements;
            
            // Calculate pointer to the start of the tail
            int16_t* d_tail_ptr = d_input_base + used_elements;
            
            CUDA_CHECK(cudaMemset(d_tail_ptr, 0, tail_elements * sizeof(int16_t)));
        }

        // Set Public Pointers
        input_valid = d_input_base + halo_elements;
        output = d_output_base;
    }

    // 2. DESTRUCTOR (Replaces free_gpu_resources)
    // Automatically called when the object goes out of scope
    ~AveragerWorkspace() {
        if (d_input_base) cudaFree(d_input_base);
        if (d_output_base) cudaFree(d_output_base);
    }

    private:
    static size_t calculateHalo(int grade, int channels, VecMode mode) {
        size_t raw_needed = grade * channels;
        // If mode is Int2 (4 elements), we align halo to multiple of 4.
        return alignUp(raw_needed, static_cast<size_t>(mode));
    }

    static size_t calculateTotal(size_t samples, size_t halo, VecMode mode) {
        size_t raw_total = halo + samples;
        // Align total size so the last vector load doesn't crash
        return alignUp(raw_total, static_cast<size_t>(mode));
    }
};