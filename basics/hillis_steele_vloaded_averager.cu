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


using namespace std;


__global__
void uniform_add(const int total_samples, const int number_of_channels, const int64_t* __restrict__ aux, int64_t* __restrict__ samples){
    int bid = blockIdx.x;
    int dim = blockDim.x;
    int tid = threadIdx.x;
    uint32_t g_index = (uint32_t)bid * dim + tid;

    if(g_index >= total_samples) return;

    if(bid > 0){
        longlong2* vec_samples = reinterpret_cast<longlong2*>(samples);
        longlong2 val = vec_samples[g_index];

        int ch1 = (g_index * 2) % number_of_channels;
        int ch2 = (g_index * 2 + 1) % number_of_channels;
        
        // Fetch Aux
        int64_t add1 = aux[(bid - 1) * number_of_channels + ch1];
        int64_t add2 = aux[(bid - 1) * number_of_channels + ch2];
        
        val.x += add1;
        val.y += add2;
        
        vec_samples[g_index] = val;
    }
        
}

// int64 for processed as it'll contain the whole sum [this'll NEVER overflow]
// UPDATE: changed samples to int64 as this will be called recursively and aux will contain large sums
__global__
void hillis_steele(const uint32_t total_elements, const int number_of_channels, const int64_t* __restrict__ samples, 
    int64_t* processedSamples, int64_t* __restrict__ aux_in_global_memory){
    extern __shared__ int64_t shared_memory[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int dim = blockDim.x;
    uint32_t g_index = (uint32_t)bid*dim + tid;

    int effective_block_size = dim * 2;
    const longlong2* vec_input = reinterpret_cast<const longlong2*>(samples);

    if (g_index < (total_elements / 2)) {
        longlong2 loaded = vec_input[g_index];
        shared_memory[2 * tid]     = loaded.x;
        shared_memory[2 * tid + 1] = loaded.y;
    } else {
        // Handle padding edge case
        shared_memory[2 * tid]     = 0;
        shared_memory[2 * tid + 1] = 0;
    }

    __syncthreads();

    for(int stride = number_of_channels; stride < effective_block_size; stride <<= 1){
        int64_t val1;
        int64_t val2;
        int idx1 = 2 * tid;
        int idx2 = 2 * tid + 1;

        if(idx1 >= stride) val1 = shared_memory[idx1 - stride];
        if(idx2 >= stride) val2 = shared_memory[idx2 - stride];

        __syncthreads();

        if(idx1 >= stride) shared_memory[idx1] += val1;
        if(idx2 >= stride) shared_memory[idx2] += val2;

        __syncthreads();
    }

    // vectorized storing back to global memory (processedSamples)
    if(g_index < total_elements / 2){
        longlong2 result;
        result.x = shared_memory[2 * tid];
        result.y = shared_memory[2 * tid + 1];
        longlong2* vec_output = reinterpret_cast<longlong2*>(processedSamples);
        vec_output[g_index] = result;
    }

    if(aux_in_global_memory != NULL && tid < number_of_channels){
        
        // A. Calculate where the sum for this channel lives in Shared Memory
        // It is located at the very end of the double-sized buffer.
        // Example (Stereo, Dim 256): Size 512. Ch0 at 510, Ch1 at 511.
        int smem_idx = (2 * dim) - number_of_channels + tid;

        // B. Calculate where to write in Global Aux
        uint32_t aux_index = (bid * number_of_channels) + tid;

        // C. Write
        aux_in_global_memory[aux_index] = shared_memory[smem_idx];
    }

}


void recursive_hillis_steele(const int block_size, const uint32_t total_samples, 
    const int number_of_channels, int64_t* samples, int64_t* aux)
{
    // 1. Calculate effective capacity per block (Vectorized)
    int items_per_block = block_size * 2; 
    uint32_t number_of_blocks = (total_samples + items_per_block - 1) / items_per_block;
    // Shared memory size is based on ITEMS, not threads
    size_t smem = items_per_block * sizeof(int64_t);
    // 2. Base Case
    if(number_of_blocks == 1){
        hillis_steele<<<1, block_size, smem>>>(
            total_samples, number_of_channels, samples, samples, NULL
        );
        return;
    }
    int64_t* current_level_aux = aux;
    int64_t* next_level_aux = aux + number_of_blocks * number_of_channels;
    // 3. Phase 1: Scan & Export Aux (Vectorized)
    hillis_steele<<<number_of_blocks, block_size, smem>>>(
        total_samples, number_of_channels, samples, samples, current_level_aux
    );
    // 4. Phase 2: Recursion 
    // We scan the aux array. Since the aux array layout is standard int64, 
    // we can use the same vectorized recursion on it!
    recursive_hillis_steele(
        block_size, 
        number_of_blocks * number_of_channels, 
        number_of_channels, 
        current_level_aux, 
        next_level_aux
    );

    // 5. Phase 3: Uniform Add (Vectorized)
    uint32_t total_vectors = (total_samples + 1) / 2; 
    uniform_add<<<number_of_blocks, block_size>>>(
        total_vectors, number_of_channels, current_level_aux, samples
    );
}


__global__
void averager_kernel(const int grade, const float inverseGrade, const int halo, const int numOfChannels, const int N, 
                            const int64_t* __restrict__ samples, int16_t* __restrict__ processedSamples){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        // 1. Get the Cumulative Sum at the current point
        // DspWorkspace guarantees that if we read past the bounds, it's handled safely, 
        // but here we are reading strictly inside 'N'.
        int64_t current_sum = samples[idx];
        
        // 2. Get the Cumulative Sum from 'grade' steps ago
        int64_t prev_sum = samples[idx - halo];

        // 3. O(1) Calculation: The sum of the window is the difference
        int64_t window_sum = current_sum - prev_sum;

        // 4. Multiply by 1/Grade to get Average
        processedSamples[idx] = static_cast<int16_t>(window_sum * inverseGrade);
    }
}

template <typename T, MemoryMode Mode>
void hillisSteeleAveragerGpuLoad(const DspWorkspace<T, Mode>& workspace, const int grade, const int blockSize, const int numOfChannels, GpuTimer& t, const vector<int64_t>& samples, vector<int16_t>& processedSamples){
    t.start();
    int64_t *d_samples = workspace.input_valid;
    int16_t *d_processedSamples = reinterpret_cast<int16_t*>(workspace.output);

    size_t totalSamples = samples.size();
    int halo = grade * numOfChannels;
    //move samples from host to device memory:
    // 2. copy samples to device memory
    MemoryTraits<Mode>::copyH2D(
        d_samples, 
        samples.data(), 
        totalSamples * sizeof(T)
    );
    
    t.mark_h2d();
    //calculate the total aux array size to allocate in one malloc command

    int64_t* scratchPad = workspace.scratchpad;
    recursive_hillis_steele(blockSize, totalSamples, numOfChannels, d_samples, scratchPad); // no need to run scan on first halo zeros
    
    
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    float inverseGrade = 1.0f / static_cast<double>(grade);
    averager_kernel<<<gridSize, blockSize>>>(grade, inverseGrade, halo, numOfChannels, totalSamples, d_samples, d_processedSamples);
    t.mark_compute();

    //move samples from device to host memory
        //move samples from device to host memory
    MemoryTraits<Mode>::copyD2H(
        processedSamples.data(), 
        d_processedSamples, 
        totalSamples * sizeof(int16_t)
    );
    
    t.stop();
}

void hillisSteeleProfiler(const int numOfChannels, const int grade, const int blockSize, 
                                                const vector<int64_t>& samples, vector<int16_t>& processedSamples){
    cout << "\n--- MEM MODE: STANDARD (Discrete) ---" << endl;
    // CPU benchmarking (cudaMalloc and cudaFree)
    ProfileResult init_res = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        size_t scratchItems = DspWorkspace<int64_t,MemoryMode::Standard>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
        DspWorkspace<int64_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Int2, scratchItems);
        t.stop();
    });

    size_t scratchItems = DspWorkspace<int64_t, MemoryMode::Standard>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
    DspWorkspace<int64_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Int2, scratchItems);

    ProfileResult process_res = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace. 
        hillisSteeleAveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int64_t), sizeof(int16_t));

    cout << "\n--- MODE: UNIFIED (Zero-Copy) ---" << endl;
    ProfileResult init_res_uni = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        size_t scratchItems_uni = DspWorkspace<int64_t,MemoryMode::Unified>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
        DspWorkspace<int64_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Int2, scratchItems_uni);
        t.stop();
    });

    size_t scratchItems_uni = DspWorkspace<int64_t, MemoryMode::Unified>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
    DspWorkspace<int64_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Int2, scratchItems_uni);

    ProfileResult process_res_uni = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace. 
        hillisSteeleAveragerGpuLoad(workspace_uni, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res_uni.initialization_ms = init_res_uni.compute_ms;
    process_res_uni.print_stats(samples.size(), sizeof(int64_t), sizeof(int16_t));

}

int32_t averager(const string pathName, const int blockSize, int point){
    vector<int64_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples64(pathName);
    if(samples.empty())
        return -1; // error in reading samples
    uint32_t totalSamples = samples.size();
    vector<int16_t> processedSamples(totalSamples);
    
    hillisSteeleProfiler(header.numChannels, point, blockSize, samples, processedSamples);

    writeSamples("hillis_steele.wav" ,header, processedSamples);
    return totalSamples;
}

int main(){
    string pathName;
    #ifdef __APPLE__
    cout<<"Enter Path to the wav file: ";
    cin>> pathName;
    #else
    pathName = "/home/nvidia/storage/DataProcessingAlgos/basics/BlueGreenRed.wav";
    #endif
    int point;
    cout<< "Enter grade for moving averager: ";
    cin>> point;
    cout<<"Enter block size: ";
    int blockSize;
    cin>>blockSize;
    if(blockSize < 32 || blockSize > 1024 || blockSize%32 != 0){
        cout<<"blockSize should be multiple of 32, >=32 && <=1024"<< endl;
        return -1;
    }

    int32_t numOfSamples = averager(pathName, blockSize, point);

    if(numOfSamples <= 0)
        cout<< "something weird happened, code: " << numOfSamples << endl;


    return 0;
}



// as shared memory size is dependenot on grade, the max grade we can test here is about 6000 if 
//  we want at least 2 blocks in the SM, or around 11700 for 1 block