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
void blelloch_uniform_add(
    const int total_frames, 
    const int frames_per_block, 
    const int64_t* __restrict__ aux, 
    int64_t* __restrict__ samples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index is FRAME
    if (idx >= total_frames) return;

    // Identify Scan Block ID
    int scan_block_id = idx / frames_per_block;

    if (scan_block_id > 0) {
        const longlong2* vec_aux = reinterpret_cast<const longlong2*>(aux);
        longlong2* vec_samples = reinterpret_cast<longlong2*>(samples);
        // Read Aux for this block (contains Left+Right sums)
        longlong2 val_to_add = vec_aux[scan_block_id - 1];
        // Add
        vec_samples[idx].x += val_to_add.x;
        vec_samples[idx].y += val_to_add.y;
    }
}

// int64 for processed as it'll contain the whole sum [this'll NEVER overflow]
// INPORTANT: ONLY VALID FOR STEREO
__global__
void blelloch_scan_inclusive(
    const uint32_t total_frames, // Note: Frames, not total samples!
    const int64_t* __restrict__ input_data, 
    int64_t* processedSamples, 
    int64_t* __restrict__ aux)
{
    // Shared Memory holds longlong2 vectors (Frames)
    extern __shared__ longlong2 vec_temp[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int dim = blockDim.x;

    // Cast pointers to vector type
    const longlong2* vec_input = reinterpret_cast<const longlong2*>(input_data);
    longlong2* vec_output      = reinterpret_cast<longlong2*>(processedSamples);
    longlong2* vec_aux         = reinterpret_cast<longlong2*>(aux);

    // Each thread processes 2 FRAMES
    int global_block_offset = bid * dim * 2; // Offset in Frames
    
    int idx_a = 2 * tid;
    int idx_b = 2 * tid + 1;
    
    uint32_t g_idx_a = global_block_offset + idx_a;
    uint32_t g_idx_b = global_block_offset + idx_b;

    // 1. Vectorized Load (Loads Left & Right simultaneously)
    if (g_idx_a < total_frames) vec_temp[idx_a] = vec_input[g_idx_a];
    else vec_temp[idx_a] = {0, 0};

    if (g_idx_b < total_frames) vec_temp[idx_b] = vec_input[g_idx_b];
    else vec_temp[idx_b] = {0, 0};

    __syncthreads();

    // 2. Vectorized Up-Sweep
    // Logic is identical to scalar, but operates on vectors
    int offset = 1;
    for (int d = dim; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            // SIMD Add: .x+=.x and .y+=.y implicitly
            longlong2 a = vec_temp[ai];
            longlong2 b = vec_temp[bi];
            
            vec_temp[bi].x = b.x + a.x;
            vec_temp[bi].y = b.y + a.y;
        }
        offset *= 2;
    }

    // 3. Write Aux
    if (tid == 0) { 
        int root_idx = (2 * dim) - 1;
        if (vec_aux != NULL) {
            // Write Total Sum (Left+Right) to Aux
            vec_aux[bid] = vec_temp[root_idx];
        }
        vec_temp[root_idx] = {0, 0}; 
    }

    // 4. Vectorized Down-Sweep
    for (int d = 1; d <= dim; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            longlong2 t = vec_temp[ai];
            vec_temp[ai] = vec_temp[bi];
            
            // SIMD Add/Swap
            longlong2 b = vec_temp[bi];
            vec_temp[bi].x = b.x + t.x;
            vec_temp[bi].y = b.y + t.y;
        }
    }
    __syncthreads();

    // 5. Store (Inclusive Convert)
    if (g_idx_a < total_frames) {
        longlong2 orig = vec_input[g_idx_a];
        longlong2 exc  = vec_temp[idx_a];
        vec_output[g_idx_a] = {exc.x + orig.x, exc.y + orig.y};
    }
    if (g_idx_b < total_frames) {
        longlong2 orig = vec_input[g_idx_b];
        longlong2 exc  = vec_temp[idx_b];
        vec_output[g_idx_b] = {exc.x + orig.x, exc.y + orig.y};
    }
}


// N is the samples per channel, not total samples
// We don't need cudaDeviceSynchronize() after kernel calls as they're executed in order they're called

// HOST: Recursive Blelloch Logic (Handles Stereo Optimization)
void recursive_blelloch(const int block_size, const uint32_t total_samples, 
    const int number_of_channels, int64_t* samples, int64_t* aux)
{
    // 1. Calculate Logical Dimensions
    // Blelloch processes 2 logical frames per thread regardless of vectorization
    int frames_per_block = block_size * 2;
    
    // Total Frames (Time steps)
    uint32_t total_frames = (total_samples + number_of_channels - 1) / number_of_channels;
    uint32_t number_of_blocks = (total_frames + frames_per_block - 1) / frames_per_block;

    // 2. Setup Aux Pointers for Recursion
    int64_t* current_level_aux = aux;
    // Next level starts after the current level's block sums
    int64_t* next_level_aux = aux + (number_of_blocks * number_of_channels);

    // PATH A: STEREO OPTIMIZATION (Vectorized int2)
    // Shared Mem: 2 frames per thread * sizeof(longlong2)
    // 1 longlong2 holds Left+Right for one frame
    size_t smem = (2 * block_size) * sizeof(longlong2);

    // A1. SCAN
    // Note: We pass NULL for aux if this is the base case (1 block)
    blelloch_scan_inclusive<<<number_of_blocks, block_size, smem>>>(
        total_frames, 
        samples, 
        samples, 
        (number_of_blocks > 1) ? current_level_aux : NULL
    );

    // A2. RECURSE
    if (number_of_blocks > 1) {
        recursive_blelloch(
            block_size, 
            number_of_blocks * number_of_channels, // Total items in Aux
            number_of_channels,                    // Aux has same channel count
            current_level_aux, 
            next_level_aux
        );
    } else {
        return; // Base case complete
    }

    // A3. ADD (Vectorized)
    // We launch enough threads to cover all FRAMES linearly
    int threads = 256;
    int blocks = (total_frames + threads - 1) / threads;
    
    blelloch_uniform_add<<<blocks, threads>>>(
        total_frames, 
        frames_per_block, 
        current_level_aux, 
        samples
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
void blellochAveragerGpuLoad(
    const DspWorkspace<T, Mode>& workspace, 
    const int grade, 
    const int blockSize, 
    const int numOfChannels, 
    GpuTimer& t, 
    const vector<int64_t>& samples, 
    vector<int16_t>& processedSamples)
{
    t.start();

    // 1. Setup Pointers
    int64_t *d_samples = workspace.input_valid;
    // Cast output to int16 (Safe parking in int64 buffer)
    int16_t *d_processedSamples = reinterpret_cast<int16_t*>(workspace.output);

    size_t totalSamples = samples.size();
    int halo = grade * numOfChannels;

    // 2. H2D Transfer
    MemoryTraits<Mode>::copyH2D(
        d_samples, 
        samples.data(), 
        totalSamples * sizeof(T)
    );

    t.mark_h2d();

    // 3. Compute Phase A: Blelloch Prefix Sum
    // Note: Blelloch prefers Power-of-2 sizes for efficiency, but 
    // our kernel handles boundaries via 'total_elements' checks.
    recursive_blelloch(blockSize, totalSamples, numOfChannels, d_samples, workspace.scratchpad);

    // 4. Compute Phase B: O(1) Difference Kernel
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    float inverseGrade = 1.0f / static_cast<float>(grade);

    // We reuse the same averager_kernel as Hillis-Steele
    averager_kernel<<<gridSize, blockSize>>>(
        grade, inverseGrade, halo, numOfChannels, totalSamples, 
        d_samples, d_processedSamples
    );
    
    t.mark_compute();

    // 5. D2H Transfer
    MemoryTraits<Mode>::copyD2H(
        processedSamples.data(), 
        d_processedSamples, 
        totalSamples * sizeof(int16_t)
    );

    
    t.stop();
}

void blellochAveragerProfiler(const int numOfChannels, const int grade, const int blockSize, 
                                                const vector<int64_t>& samples, vector<int16_t>& processedSamples){
    // CPU benchmarking (cudaMalloc and cudaFree)
    cout << "\n--- MEM MODE: STANDARD (Discrete) ---" << endl;
    ProfileResult init_res = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        size_t scratchItems = DspWorkspace<int64_t,MemoryMode::Standard>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
        DspWorkspace<int64_t,MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Scalar, scratchItems);
        t.stop();
    });

    size_t scratchItems = DspWorkspace<int64_t,MemoryMode::Standard>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
    DspWorkspace<int64_t,MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Scalar, scratchItems);

    ProfileResult process_res = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace. 
        blellochAveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int64_t), sizeof(int16_t));

    cout << "\n--- MODE: UNIFIED (Zero-Copy) ---" << endl;

    ProfileResult init_res_uni = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        size_t scratchItems_uni = DspWorkspace<int64_t,MemoryMode::Unified>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
        DspWorkspace<int64_t,MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Scalar, scratchItems_uni);
        t.stop();
    });

    size_t scratchItems_uni = DspWorkspace<int64_t,MemoryMode::Unified>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
    DspWorkspace<int64_t,MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Scalar, scratchItems_uni);

    ProfileResult process_res_uni = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace. 
        blellochAveragerGpuLoad(workspace_uni, grade, blockSize, numOfChannels, t, samples, processedSamples);
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
    
    blellochAveragerProfiler(header.numChannels, point, blockSize, samples, processedSamples);

    writeSamples("blelloch_averager.wav" ,header, processedSamples);
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

// Current limitation: blockSize % channels should be 0