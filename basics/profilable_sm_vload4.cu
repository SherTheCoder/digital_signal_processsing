#include<iostream>
#include<fstream>
#include <vector>
#include <string>
#include <tuple>
#include<chrono>
#include<cuda_runtime.h>
#include "../wav_header.h"
#include "../benchmark.h"
#include "../gpu_utils.h"

// grade - 1 is the number of duplicate calls we make to the VRAM per block
// the larger the grade, the higher the duplicate calls

// the larger the block size, the fewer the duplicate calls, but higher occupancy issues

// Nice problem : find the optimal block size given a grade
// for small grade, a small block size is good
// for large grade, we might get to a point where the duplicate calls per block to VRAM exceed the 
//      occupancy advantage of a small block

/*
Choosing padding over discarding might seem like a speed-memory tradeoff, but it's not.
Using padding NEVER results in extra shared memory being assigned, as sm is always assigned in 
multiples of 256 bytes in Maxwell [Jetson Nano]. So we fill the last couplt of spots on sm that we don't use, 
but they were allocated anyway
*/

using namespace std;

//__restrict__ means no other pointer aliases with this one, helps compiler optimize memory accesses
__global__
void averager_kernel(
    const int grade, 
    const float inverseGrade, 
    const int numOfChannels, 
    const int N, 
    const int16_t* __restrict__ samples, 
    int16_t* __restrict__ processedSamples, 
    const int halo)
{
    extern __shared__ int16_t shared_memory[];
    const int4* vec_samples = reinterpret_cast<const int4*>(samples);
    int blockSize = blockDim.x;
    int total_vectors = (blockSize + halo) / 8;
    int start_vec_idx = ((blockIdx.x * blockSize) - halo) / 8;
    int max_vectors = (N + 7) / 8;

    for(int i = threadIdx.x; i < total_vectors; i += blockSize){
        
        int sm_idx = i * 8;
        int current_vec_idx = start_vec_idx + i;
        
        if (current_vec_idx < max_vectors){
            int4 vector_data = vec_samples[start_vec_idx + i]; 
            // -- Unpack .x --
            shared_memory[sm_idx]     = (int16_t)(vector_data.x & 0xFFFF);       
            shared_memory[sm_idx + 1] = (int16_t)((vector_data.x >> 16) & 0xFFFF); 
            // -- Unpack .y --
            shared_memory[sm_idx + 2] = (int16_t)(vector_data.y & 0xFFFF);       
            shared_memory[sm_idx + 3] = (int16_t)((vector_data.y >> 16) & 0xFFFF); 
            // -- Unpack .z --
            shared_memory[sm_idx + 4] = (int16_t)(vector_data.z & 0xFFFF);       
            shared_memory[sm_idx + 5] = (int16_t)((vector_data.z >> 16) & 0xFFFF); 
            // -- Unpack .w --
            shared_memory[sm_idx + 6] = (int16_t)(vector_data.w & 0xFFFF);       
            shared_memory[sm_idx + 7] = (int16_t)((vector_data.w >> 16) & 0xFFFF);   
        }
        else{
                shared_memory[sm_idx]     = 0;       
                shared_memory[sm_idx + 1] = 0; 
                // -- Unpack .y --
                shared_memory[sm_idx + 2] = 0;       
                shared_memory[sm_idx + 3] = 0; 
                // -- Unpack .z --
                shared_memory[sm_idx + 4] = 0;       
                shared_memory[sm_idx + 5] = 0; 
                // -- Unpack .w --
                shared_memory[sm_idx + 6] = 0;       
                shared_memory[sm_idx + 7] = 0;  
        }
 
    }
    
    __syncthreads();

    // 5. COMPUTE (Standard Sliding Window)
    int g_threadIndex = (blockIdx.x * blockSize) + threadIdx.x;
    if(g_threadIndex < N){
        int32_t sum = 0;
        // Pointer math optimization: Point to "My Window"
        const int16_t* my_window = &shared_memory[halo + threadIdx.x];
        #pragma unroll
        for(int i = 0 ; i < grade; i++)
            sum += my_window[-(i * numOfChannels)];
        processedSamples[g_threadIndex] = static_cast<int16_t>(sum * inverseGrade); 
    }
}

template <typename T, MemoryMode Mode>
void vload4AveragerGpuLoad(const DspWorkspace<T, Mode>& workspace, const int grade, const int blockSize, const int numOfChannels, GpuTimer& t, const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    t.start();
    int16_t *d_samples = workspace.input_valid;
    int16_t *d_processedSamples = workspace.output;

    int totalSamples = samples.size();
    int halo = static_cast<int>(workspace.halo_elements);
    //move samples from host to device memory:
    // copy samples to device memory
    MemoryTraits<Mode>::copyH2D(
        d_samples, 
        samples.data(), 
        totalSamples * sizeof(int16_t)
    );

    // move complete
    t.mark_h2d();    
    
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    // IMPORTANT: we are upsizing the shared memrory to hold a number of samples that is multiple of 8.
    /*Yes, we could choose to not do it, and populate shared memory based on conditions after loading 
    the int2 from VRAM. But that would add extra instructions and hurt performance.
        example:

        if(sm_base_idx + 1 < numberOfSamplesInSharedMemory)
            shared_memory[sm_base_idx + 1] = (int16_t)((vector_data.x >> 16) & 0xFFFF); 
        if(sm_base_idx + 2 < numberOfSamplesInSharedMemory)
            shared_memory[sm_base_idx + 2] = (int16_t)(vector_data.y & 0xFFFF);      
        if(sm_base_idx + 3 < numberOfSamplesInSharedMemory)
            shared_memory[sm_base_idx + 3] = (int16_t)((vector_data.y >> 16) & 0xFFFF); 

    So we choose PADDING > BRANCHING for better performance.
    Here's the best part : PADDING DOES NOT ADD A MEMORY OVERHEAD AS SHARED MEMORY IS ALLOCATED BY GPU 
        IN CHUNKS OF 256 BYTES [Maxwell Architexture of Jetson Nano].
        So if halo + blockSize = 257 bytes, it'll allocate 512 bytes.
        There's no situation in which the shared memory allocation increases because of padding. 
        It only increases based on the size of Halo, as
            1. block size is always a mutliple of 32 (which is the warp size)
            2. we only pad to get to the next multiple of 4, which is going to happen anyway. 
    */

    int numberOfSamplesInSharedMemory = (blockSize + halo);
    int sharedMemorySize = numberOfSamplesInSharedMemory * sizeof(int16_t);
    float inverseGrade = 1.0f / static_cast<float>(grade);
    averager_kernel<<<gridSize, blockSize, sharedMemorySize>>>(grade, inverseGrade, numOfChannels, totalSamples, d_samples, d_processedSamples, halo);
    t.mark_compute();
    //move samples from device to host memory
    MemoryTraits<Mode>::copyD2H(
        processedSamples.data(), 
        d_processedSamples, 
        totalSamples * sizeof(int16_t)
    );

    t.stop();
}

void vload4AveragerProfiler(const int numOfChannels, const int grade, const int blockSize, 
                                                const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    CsvLogger logger("benchmark_data.csv");
    cout << "\n--- MEM MODE: STANDARD (Discrete) ---" << endl;
    // CPU benchmarking (cudaMalloc and cudaFree)
    ProfileResult init_res = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        // Constructor runs cudaMalloc
        DspWorkspace<int16_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Int4); 
        t.stop();
        // Destructor runs cudaFree AUTOMATICALLY here (end of scope)
    });
    DspWorkspace<int16_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Int4); 

    // GPU benchmarking (cudaMemcpy from host -> kernel execution -> cudaMemcpy to host)
    ProfileResult process_res = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace object
        vload4AveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int16_t));

    logger.log(
        "Vectorized SM4 Parallel",      // Algorithm Name
        "Standard",          // Mode
        samples.size(),      // N
        grade,               // Grade
        blockSize,           // Block Size
        process_res,         // The Results
        sizeof(int16_t)      // Input Size
    );

    cout << "\n--- MODE: UNIFIED (Zero-Copy) ---" << endl;

    ProfileResult init_res_uni = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        // Constructor runs cudaMalloc
        DspWorkspace<int16_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Int4); 
        t.stop();
        // Destructor runs cudaFree AUTOMATICALLY here (end of scope)
    });
    DspWorkspace<int16_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Int4); 

    // GPU benchmarking (cudaMemcpy from host -> kernel execution -> cudaMemcpy to host)
    ProfileResult process_res_uni = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace object
        vload4AveragerGpuLoad(workspace_uni, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res_uni.initialization_ms = init_res_uni.compute_ms;
    process_res_uni.print_stats(samples.size(), sizeof(int16_t));

    logger.log(
        "Vectorized SM4 Parallel",      // Algorithm Name
        "Unified",          // Mode
        samples.size(),      // N
        grade,               // Grade
        blockSize,           // Block Size
        process_res_uni,         // The Results
        sizeof(int16_t)      // Input Size
    );
   
}

int32_t averager(const string pathName, const int blockSize, int point){
    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty())
        return -1; // error in reading samples
    uint32_t totalSamples = samples.size();
    vector<int16_t> processedSamples(totalSamples);
    
    vload4AveragerProfiler(header.numChannels, point, blockSize, samples, processedSamples);
    writeSamples("profile_sm_averager.wav" ,header, processedSamples);
    return totalSamples;
}

int main(int argc, char* argv[]) {
    // Usage: ./exe <path> <grade> <blockSize>
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <wav_path> <grade> <block_size>" << std::endl;
        return 1;
    }

    std::string pathName = argv[1];
    int grade = std::stoi(argv[2]);
    int blockSize = std::stoi(argv[3]);

    // Validation
    if(blockSize < 32 || blockSize > 1024 || blockSize % 32 != 0){
        std::cerr << "Error: Block size must be multiple of 32" << std::endl;
        return 1; 
    }

    // Calling averager function
    uint32_t result = averager(pathName, blockSize, grade);

    return (result > 0) ? 0 : 1;
}

