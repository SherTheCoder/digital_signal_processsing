#pragma once
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include "wav_header.h"


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

enum class MemoryMode {
    Standard, // cudaMalloc + cudaMemcpy
    Unified   // cudaMallocManaged + std::memcpy (Zero Copy)
};

// 2. The Policy Handler (Traits)
// This struct handles the logic differences so the main code doesn't have to.
template <MemoryMode Mode>
struct MemoryTraits;

// -- Specialization for STANDARD --
template <>
struct MemoryTraits<MemoryMode::Standard> {
    static void allocate(void** ptr, size_t size) {
        CUDA_CHECK(cudaMalloc(ptr, size));
    }
    
    static void copyH2D(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
    
    static void copyD2H(void* dst, const void* src, size_t size) {
        // cudaMemcpy is implicit sync
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
};

// -- Specialization for UNIFIED --
template <>
struct MemoryTraits<MemoryMode::Unified> {
    static void allocate(void** ptr, size_t size) {
        // Managed memory is accessible by both CPU and GPU
        CUDA_CHECK(cudaMallocManaged(ptr, size));
    }
    
    static void copyH2D(void* dst, const void* src, size_t size) {
        // CPU writes directly to the pointer
        std::memcpy(dst, src, size);
    }
    
    static void copyD2H(void* dst, const void* src, size_t size) {
        // CRITICAL: We must sync before CPU reads results!
        CUDA_CHECK(cudaDeviceSynchronize()); // This might slow things down and prevent latency hiding
        std::memcpy(dst, src, size);
    }
};

template <typename T, MemoryMode Mode = MemoryMode::Standard>
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
        MemoryTraits<Mode>::allocate((void**)&d_input_base, total_input_bytes);
        
        // 2. Allocate Output
        MemoryTraits<Mode>::allocate((void**)&d_output_base, valid_bytes);

        // 3. Allocate Scratchpad (Recursive Block Sums)
        if (scratch_bytes > 0) {
            MemoryTraits<Mode>::allocate((void**)&d_scratch_base, scratch_bytes);
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

class CsvLogger {
private:
    std::string filename;
    
    // Helper to check if file exists (to decide whether to write header)
    bool fileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

public:
    CsvLogger(const std::string& fname = "benchmark_results.csv") : filename(fname) {}

    void log(const std::string& algo_name, 
             const std::string& memory_mode, 
             size_t N, 
             int grade, 
             int block_size, 
             const ProfileResult& res,
             size_t input_size_bytes, 
             size_t output_size_bytes = 0) 
    {
        // Default output size to input size if not provided
        if (output_size_bytes == 0) output_size_bytes = input_size_bytes;

        bool isNewFile = !fileExists(filename);
        
        // Open file in Append Mode
        std::ofstream file;
        file.open(filename, std::ios::app);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open CSV file " << filename << std::endl;
            return;
        }

        // 1. Write Header (If new file)
        if (isNewFile) {
            file << "Algorithm,MemoryMode,N_Samples,Grade,BlockSize,"
                 << "H2D_ms,Compute_ms,D2H_ms,Total_ms,"
                 << "Init_ms,ColdStart_Total_ms,"
                 << "Bandwidth_GBs,Throughput_MSs,ColdStart_MSs\n";
        }

        // 2. Calculate Metrics (Same math as print_stats)
        double total_bytes = static_cast<double>(N) * (input_size_bytes + output_size_bytes);
        double total_gb = total_bytes / 1e9;
        double total_samples_mega = N / 1e6;

        double steady_sec = res.total_ms / 1000.0;
        double cold_sec = (res.initialization_ms + res.total_ms) / 1000.0;

        double bandwidth = (steady_sec > 0) ? (total_gb / steady_sec) : 0.0;
        double throughput = (steady_sec > 0) ? (total_samples_mega / steady_sec) : 0.0;
        double cold_throughput = (cold_sec > 0) ? (total_samples_mega / cold_sec) : 0.0;

        // 3. Write Data Row
        file << algo_name << ","
             << memory_mode << ","
             << N << ","
             << grade << ","
             << block_size << ","
             << res.transfer_h2d_ms << ","
             << res.compute_ms << ","
             << res.transfer_d2h_ms << ","
             << res.total_ms << ","
             << res.initialization_ms << ","
             << (res.initialization_ms + res.total_ms) << "," // Cold Start Latency
             << bandwidth << ","
             << throughput << ","
             << cold_throughput << "\n";

        file.close();
        std::cout << ">> Data saved to " << filename << std::endl;
    }
};