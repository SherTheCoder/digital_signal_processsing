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
        int channel = g_index % number_of_channels;
        uint32_t aux_index = (uint32_t)(bid - 1) * number_of_channels + channel;
        samples[g_index] += aux[aux_index];
    }
        
}

__global__
void hillis_steele(const uint32_t total_elements, const int number_of_channels, const int64_t* __restrict__ samples, 
    int64_t* processedSamples, int64_t* __restrict__ aux_in_global_memory){
    extern __shared__ int64_t shared_memory[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int dim = blockDim.x;
    uint32_t g_index = (uint32_t)bid*dim + tid;

    if(g_index < total_elements)
        shared_memory[tid] = samples[g_index];
    else
        shared_memory[tid] = 0;
    __syncthreads();

    for(int stride = number_of_channels; stride < dim; stride <<= 1){
        int64_t val;
        if(tid >= stride)
            val = shared_memory[tid - stride];
        __syncthreads();
        if(tid >= stride)
            shared_memory[tid] += val;
        __syncthreads();
    }

    if(g_index < total_elements)
        processedSamples[g_index] = shared_memory[tid];

    if(aux_in_global_memory != NULL && tid >= dim - number_of_channels && tid < dim){
        int channel = tid % number_of_channels;
        uint32_t aux_index = (uint32_t)bid * number_of_channels + channel;
        aux_in_global_memory[aux_index] = shared_memory[tid];
    }

}

void recursive_hillis_steele(const int block_size, const uint32_t total_samples, 
    const int number_of_channels, int64_t*  samples, int64_t* aux){
    uint32_t number_of_blocks = (uint32_t)(total_samples + block_size - 1) / block_size;
    if(number_of_blocks == 1){
        size_t size_of_shared_memory = total_samples * sizeof(int64_t);
        hillis_steele<<<1, total_samples, size_of_shared_memory>>>(total_samples, number_of_channels, samples, samples, NULL);
        return;
    }
    int64_t* current_level_aux = aux;
    int64_t* next_level_aux = aux + number_of_blocks * number_of_channels;
    size_t size_of_shared_memory = block_size * sizeof(int64_t);
    hillis_steele<<< number_of_blocks, block_size, size_of_shared_memory>>>(total_samples, number_of_channels, samples, samples, current_level_aux);
    recursive_hillis_steele(block_size, number_of_blocks * number_of_channels, number_of_channels, current_level_aux, next_level_aux);
    uniform_add<<<number_of_blocks, block_size>>>(total_samples, number_of_channels, current_level_aux, samples);

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
void hillisSteeleAveragerGpuLoad(const DspWorkspace<T, Mode>& workspace, const int grade, const int blockSize, const int numOfChannels, GpuTimer& t, const vector<int64_t>& samples, vector<int16_t>& processedSamples){
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
    int64_t* scratchPad = workspace.scratchpad;
    recursive_hillis_steele(blockSize, totalSamples, numOfChannels, d_samples, scratchPad);
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    float inverseGrade = 1.0f / static_cast<double>(grade);
    averager_kernel<<<gridSize, blockSize>>>(grade, inverseGrade, halo, numOfChannels, totalSamples, d_samples, d_processedSamples);
    t.mark_compute();
    MemoryTraits<Mode>::copyD2H(
        processedSamples.data(), 
        d_processedSamples, 
        totalSamples * sizeof(int16_t)
    );
    
    t.stop();
}

void hillisSteeleProfiler(const int numOfChannels, const int grade, const int blockSize, 
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
        hillisSteeleAveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int64_t), sizeof(int16_t));

    logger.log(
        "HillisSteele",
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
        hillisSteeleAveragerGpuLoad(workspace_uni, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res_uni.initialization_ms = init_res_uni.compute_ms;
    process_res_uni.print_stats(samples.size(), sizeof(int64_t), sizeof(int16_t));

    logger.log(
        "HillisSteele", 
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
    cout<<"--- Hillis Steele Averager ---" << endl;
    cout<<"total samples: " << totalSamples << endl;
    cout<< "point: " << point << endl;

    hillisSteeleProfiler(header.numChannels, point, blockSize, samples, processedSamples);
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