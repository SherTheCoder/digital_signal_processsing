#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
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
    Scalar = 1, // Standard T access
    Int2   = 4, // Loading 4 shorts (8 bytes) / 2 Int64 (16 bytes)
    Int4   = 8  // Loading 8 shorts (16 bytes) / 4 Int64 (32 bytes)
};

template <typename T>
class DspWorkspace {
private:
    T* d_input_base = nullptr;  
    T* d_output_base = nullptr; 
    T* d_scratch_base = nullptr;

    // Helper: Align up to nearest multiple
    static size_t alignUp(size_t x, size_t alignment) {
        return (x + alignment - 1) & ~(alignment - 1);
    }

    // Disable Copying
    DspWorkspace(const DspWorkspace&) = delete;
    DspWorkspace& operator=(const DspWorkspace&) = delete;

public:
    // Public Accessors
    T* input_valid = nullptr; 
    T* output = nullptr;      
    T* scratchpad = nullptr;  

    const size_t halo_elements;     
    const size_t total_alloc_elements; 
    const size_t valid_bytes;
    const size_t scratch_bytes;

    // -------------------------------------------------------------------------
    // CONSTRUCTOR
    // -------------------------------------------------------------------------
    DspWorkspace(size_t num_samples, int grade, int num_channels, 
                 VecMode mode = VecMode::Scalar, 
                 size_t extra_scratch_elements = 0)
        : halo_elements(calculateHalo(grade, num_channels, mode)),
          total_alloc_elements(calculateTotal(num_samples, halo_elements, mode)),
          valid_bytes(num_samples * sizeof(T)),
          scratch_bytes(extra_scratch_elements * sizeof(T)) {

        size_t total_input_bytes = total_alloc_elements * sizeof(T);

        // 1. Allocate Input (Halo + Data + Padding)
        CUDA_CHECK(cudaMalloc(&d_input_base, total_input_bytes));
        
        // 2. Allocate Output
        CUDA_CHECK(cudaMalloc(&d_output_base, valid_bytes));

        // 3. Allocate Scratchpad (Recursive Block Sums)
        if (scratch_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&d_scratch_base, scratch_bytes));
            CUDA_CHECK(cudaMemset(d_scratch_base, 0, scratch_bytes));
            scratchpad = d_scratch_base;
        }

        // ---------------------------------------------------------
        // Zeroing Logic (Crucial for Blelloch & Vectorization)
        // ---------------------------------------------------------
        
        // A. Zero the Halo (History)
        if (halo_elements > 0) {
            CUDA_CHECK(cudaMemset(d_input_base, 0, halo_elements * sizeof(T)));
        }

        // B. Zero the Tail (Alignment Padding / Power-Of-2 Padding)
        // Blelloch kernels often run on PoT grids. If your data is 100 but you 
        // allocated 128, the kernel will read indices 100-127. 
        // Zeroing ensures these act as "Identity elements" (adding 0 changes nothing).
        size_t used_elements = halo_elements + num_samples;
        if (total_alloc_elements > used_elements) {
            size_t tail_elements = total_alloc_elements - used_elements;
            T* d_tail_ptr = d_input_base + used_elements;
            CUDA_CHECK(cudaMemset(d_tail_ptr, 0, tail_elements * sizeof(T)));
        }

        // Set Public Pointers
        input_valid = d_input_base + halo_elements;
        output = d_output_base;
    }

    // DESTRUCTOR
    ~DspWorkspace() {
        if (d_input_base) cudaFree(d_input_base);
        if (d_output_base) cudaFree(d_output_base);
        if (d_scratch_base) cudaFree(d_scratch_base);
    }

    // -------------------------------------------------------------------------
    // HELPERS
    // -------------------------------------------------------------------------
    static size_t calculateHalo(int grade, int channels, VecMode mode) {
        size_t raw = grade * channels;
        return alignUp(raw, static_cast<size_t>(mode));
    }

    static size_t calculateTotal(size_t samples, size_t halo, VecMode mode) {
        return alignUp(halo + samples, static_cast<size_t>(mode));
    }

    // Updated: Generic name for ANY recursive scan (Hillis-Steele OR Blelloch)
    static size_t calcMultiBlockScratchSize(size_t totalSamples, int blockSize, int numChannels) {
        size_t size_needed = 0;
        size_t n_blocks = (totalSamples + blockSize - 1) / blockSize;
        while(n_blocks > 1){
            // Assumes we store independent sums for each channel
            n_blocks = n_blocks * numChannels; 
            size_needed += n_blocks;
            // Next level reduction
            n_blocks = (n_blocks + blockSize - 1) / blockSize;
        }
        return size_needed;
    }

    // New: Blelloch Helper
    // Useful if you want to force your buffer size to be a Power of Two
    // Usage: DspWorkspace ws(DspWorkspace::nextPowerOfTwo(N), ...);
    static size_t nextPowerOfTwo(size_t n) {
        size_t power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
};