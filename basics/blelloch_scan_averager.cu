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
    const int total_items, 
    const int num_channels, 
    const int items_per_logical_block, 
    const int64_t* __restrict__ scanned_aux, 
    int64_t* __restrict__ data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_items) return;
    
    int frame_idx = idx / num_channels;
    int channel_idx = idx % num_channels;
        
    int scan_block_id = frame_idx / items_per_logical_block; 

    if (scan_block_id > 0) {
        int aux_idx = (scan_block_id - 1) * num_channels + channel_idx;
        data[idx] += scanned_aux[aux_idx];
    }
}

// int64 for processed as it'll contain the whole sum [this'll NEVER overflow]
__global__
void blelloch_scan_inclusive(
    const uint32_t total_elements, 
    const int num_channels, 
    const int64_t* __restrict__ input_data, 
    int64_t* processedSamples, 
    int64_t* __restrict__ aux)
{
    extern __shared__ int64_t temp[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int dim = blockDim.x;

    int global_block_offset = bid * dim * 2 * num_channels;
    
    int logical_a = 2 * tid;
    int logical_b = 2 * tid + 1;

    for(int ch = 0; ch < num_channels; ++ch) {
        int s_idx_a = (logical_a * num_channels) + ch;
        int s_idx_b = (logical_b * num_channels) + ch;
        
        uint32_t g_idx_a = global_block_offset + s_idx_a;
        uint32_t g_idx_b = global_block_offset + s_idx_b;

        if (g_idx_a < total_elements) temp[s_idx_a] = input_data[g_idx_a];
        else temp[s_idx_a] = 0;

        if (g_idx_b < total_elements) temp[s_idx_b] = input_data[g_idx_b];
        else temp[s_idx_b] = 0;
    }
    __syncthreads();

    int offset = 1;
    for (int d = dim; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            for(int ch = 0; ch < num_channels; ++ch) {
                int phys_ai = (ai * num_channels) + ch;
                int phys_bi = (bi * num_channels) + ch;
                temp[phys_bi] += temp[phys_ai];
            }
        }
        offset *= 2;
    }


    if (tid == 0) { 
        int root_idx = (2 * dim) - 1;
        for(int ch = 0; ch < num_channels; ++ch) {
            int phys_root = (root_idx * num_channels) + ch;
            if (aux != NULL) {
                aux[bid * num_channels + ch] = temp[phys_root];
            }
            temp[phys_root] = 0; 
        }
    }


    for (int d = 1; d <= dim; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            for(int ch = 0; ch < num_channels; ++ch) {
                int phys_ai = (ai * num_channels) + ch;
                int phys_bi = (bi * num_channels) + ch;
                int64_t t = temp[phys_ai];
                temp[phys_ai] = temp[phys_bi];
                temp[phys_bi] += t;
            }
        }
    }
    __syncthreads();
    
    for(int ch = 0; ch < num_channels; ++ch) {
        int s_idx_a = (logical_a * num_channels) + ch;
        int s_idx_b = (logical_b * num_channels) + ch;
        
        uint32_t g_idx_a = global_block_offset + s_idx_a;
        uint32_t g_idx_b = global_block_offset + s_idx_b;

        if (g_idx_a < total_elements) {
            processedSamples[g_idx_a] = temp[s_idx_a] + input_data[g_idx_a];
        }
        if (g_idx_b < total_elements) {
            processedSamples[g_idx_b] = temp[s_idx_b] + input_data[g_idx_b];
        }
    }
}


void recursive_blelloch(const int block_size, const uint32_t total_samples, 
    const int number_of_channels, int64_t* samples, int64_t* aux)
{
    int frames_per_block = block_size * 2;
    int items_per_block_phys = frames_per_block * number_of_channels;

    uint32_t total_frames = total_samples / number_of_channels; 
    uint32_t number_of_blocks = (total_frames + frames_per_block - 1) / frames_per_block;

    size_t smem = items_per_block_phys * sizeof(int64_t);

    if (number_of_blocks == 1) {
        blelloch_scan_inclusive<<<1, block_size, smem>>>(
            total_samples, number_of_channels, samples, samples, NULL
        );
        return;
    }

    int64_t* current_level_aux = aux;
    int64_t* next_level_aux = aux + (number_of_blocks * number_of_channels);

    blelloch_scan_inclusive<<<number_of_blocks, block_size, smem>>>(
        total_samples, number_of_channels, samples, samples, current_level_aux
    );

    recursive_blelloch(block_size, number_of_blocks * number_of_channels, number_of_channels, current_level_aux, next_level_aux);
    
    int threads_for_add = 256;
    int blocks_for_add = (total_samples + threads_for_add - 1) / threads_for_add;
    
    blelloch_uniform_add<<<blocks_for_add, threads_for_add>>>(
        total_samples, number_of_channels, frames_per_block, current_level_aux, samples
    );
}


__global__
void averager_kernel(const int grade, const float inverseGrade, const int halo, const int numOfChannels, const int N, 
                            const int64_t* __restrict__ samples, int16_t* __restrict__ processedSamples){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        int64_t current_sum = samples[idx];
        
        int64_t prev_sum = samples[idx - halo];

        int64_t window_sum = current_sum - prev_sum;

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

    int64_t *d_samples = workspace.input_valid;
    int16_t *d_processedSamples = reinterpret_cast<int16_t*>(workspace.output);

    size_t totalSamples = samples.size();
    int halo = grade * numOfChannels;

    MemoryTraits<Mode>::copyH2D(
        d_samples, 
        samples.data(), 
        totalSamples * sizeof(T)
    );

    t.mark_h2d();

    recursive_blelloch(blockSize, totalSamples, numOfChannels, d_samples, workspace.scratchpad);
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    float inverseGrade = 1.0f / static_cast<float>(grade);
    averager_kernel<<<gridSize, blockSize>>>(
        grade, inverseGrade, halo, numOfChannels, totalSamples, 
        d_samples, d_processedSamples
    );
    
    t.mark_compute();

    MemoryTraits<Mode>::copyD2H(
        processedSamples.data(), 
        d_processedSamples, 
        totalSamples * sizeof(int16_t)
    );

    
    t.stop();
}

void blellochAveragerProfiler(const int numOfChannels, const int grade, const int blockSize, 
                                                const vector<int64_t>& samples, vector<int16_t>& processedSamples){
    CsvLogger logger("benchmark_data.csv");
    cout << "\n--- MEM MODE: STANDARD (Discrete) ---" << endl;
    ProfileResult init_res = benchmark<CpuTimer>(measurementRounds, warmupRounds, [&](CpuTimer& t) {
        t.start();
        size_t scratchItems = DspWorkspace<int64_t, MemoryMode::Standard>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
        DspWorkspace<int64_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Scalar, scratchItems);
        t.stop();
    });

    size_t scratchItems = DspWorkspace<int64_t, MemoryMode::Standard>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
    DspWorkspace<int64_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Scalar, scratchItems);

    ProfileResult process_res = benchmark<GpuTimer>(measurementRounds, warmupRounds, [&](GpuTimer& t) {
        blellochAveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int64_t), sizeof(int16_t));

    logger.log(
        "Blelloch",      
        "Standard",          
        samples.size(),      
        grade,               
        blockSize,           
        process_res,         
        sizeof(int64_t),    
        sizeof(int16_t)      
    );

    cout << "\n--- MODE: UNIFIED (Zero-Copy) ---" << endl;
    ProfileResult init_res_uni = benchmark<CpuTimer>(measurementRounds, warmupRounds, [&](CpuTimer& t) {
        t.start();
        size_t scratchItems_uni = DspWorkspace<int64_t, MemoryMode::Unified>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
        DspWorkspace<int64_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Scalar, scratchItems_uni);
        t.stop();
    });

    size_t scratchItems_uni = DspWorkspace<int64_t, MemoryMode::Unified>::calcMultiBlockScratchSize(samples.size(), blockSize, numOfChannels);
    DspWorkspace<int64_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Scalar, scratchItems_uni);

    ProfileResult process_res_uni = benchmark<GpuTimer>(measurementRounds, warmupRounds, [&](GpuTimer& t) {
        // Pass the workspace. 
        blellochAveragerGpuLoad(workspace_uni, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res_uni.initialization_ms = init_res_uni.compute_ms;
    process_res_uni.print_stats(samples.size(), sizeof(int64_t), sizeof(int16_t));

    logger.log(
        "Blelloch", 
        "Unified", 
        samples.size(), 
        grade, 
        blockSize, 
        process_res_uni, 
        sizeof(int64_t), 
        sizeof(int16_t)
    );

}

int32_t averager(const string pathName, const int blockSize, int point){
    vector<int64_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples64(pathName);
    if(samples.empty())
        return -1; 
    uint32_t totalSamples = samples.size();
    vector<int16_t> processedSamples(totalSamples);
    
    blellochAveragerProfiler(header.numChannels, point, blockSize, samples, processedSamples);

    // writeSamples("blelloch_averager.wav" ,header, processedSamples);
    return totalSamples;
}

int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <wav_path> <grade> <block_size>" << std::endl;
        return 1;
    }

    std::string pathName = argv[1];
    int grade = std::stoi(argv[2]);
    int blockSize = std::stoi(argv[3]);

    if(blockSize < 32 || blockSize > 1024 || blockSize % 32 != 0){
        std::cerr << "Error: Block size must be multiple of 32" << std::endl;
        return 1; 
    }

    uint32_t result = averager(pathName, blockSize, grade);

    return (result > 0) ? 0 : 1;
}


// as shared memory size is dependenot on grade, the max grade we can test here is about 6000 if 
//  we want at least 2 blocks in the SM, or around 11700 for 1 block

// Current limitation: blockSize % channels should be 0