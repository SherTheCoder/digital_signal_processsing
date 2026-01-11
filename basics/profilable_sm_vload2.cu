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

    const int2* vec_samples = reinterpret_cast<const int2*>(samples);
    int blockSize = blockDim.x;
    int total_vectors = (blockSize + halo) / 4;
    int start_vec_idx = ((blockIdx.x * blockSize) - halo) / 4;
    int max_vectors = (N + 3) / 4;
    for(int i = threadIdx.x; i < total_vectors; i += blockSize){
        int sm_idx = i * 4;
        int current_vec_idx = start_vec_idx + i;
        if (current_vec_idx < max_vectors){
            int2 vector_data = vec_samples[start_vec_idx + i]; 
            shared_memory[sm_idx]     = (int16_t)(vector_data.x & 0xFFFF);       
            shared_memory[sm_idx + 1] = (int16_t)((vector_data.x >> 16) & 0xFFFF); 
            shared_memory[sm_idx + 2] = (int16_t)(vector_data.y & 0xFFFF);       
            shared_memory[sm_idx + 3] = (int16_t)((vector_data.y >> 16) & 0xFFFF); 
        }
        else{
            shared_memory[sm_idx] = 0;
            shared_memory[sm_idx+1] = 0;
            shared_memory[sm_idx+2] = 0;
            shared_memory[sm_idx+3] = 0;
        }
        
    }
    
    __syncthreads();

    int g_threadIndex = (blockIdx.x * blockSize) + threadIdx.x;
    if(g_threadIndex < N){
        int32_t sum = 0;
        const int16_t* my_window = &shared_memory[halo + threadIdx.x];
        #pragma unroll
        for(int i = 0 ; i < grade; i++)
            sum += my_window[-(i * numOfChannels)];
        processedSamples[g_threadIndex] = static_cast<int16_t>(sum * inverseGrade); 
    }
}

template <typename T, MemoryMode Mode>
void vload2AveragerGpuLoad(const DspWorkspace<T, Mode>& workspace, const int grade, const int blockSize, const int numOfChannels, GpuTimer& t, const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    t.start();
    int16_t *d_samples = workspace.input_valid;
    int16_t *d_processedSamples = workspace.output;

    int totalSamples = samples.size();
    int halo = static_cast<int>(workspace.halo_elements);
    MemoryTraits<Mode>::copyH2D(
        d_samples, 
        samples.data(), 
        totalSamples * sizeof(int16_t)
    );
    t.mark_h2d();    
    
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    int numberOfSamplesInSharedMemory = (blockSize + halo);
    int sharedMemorySize = numberOfSamplesInSharedMemory * sizeof(int16_t);
    float inverseGrade = 1.0f / static_cast<float>(grade);
    averager_kernel<<<gridSize, blockSize, sharedMemorySize>>>(grade, inverseGrade, numOfChannels, totalSamples, d_samples, d_processedSamples, halo);
    t.mark_compute();
    MemoryTraits<Mode>::copyD2H(
        processedSamples.data(), 
        d_processedSamples, 
        totalSamples * sizeof(int16_t)
    );

    t.stop();
}

void vload2AveragerProfiler(const int numOfChannels, const int grade, const int blockSize, 
                                                const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    CsvLogger logger("benchmark_data.csv");

    cout << "\n--- MEM MODE: STANDARD (Discrete) ---" << endl;
    ProfileResult init_res = benchmark<CpuTimer>(measurementRounds, warmupRounds, [&](CpuTimer& t) {
        t.start();
        DspWorkspace<int16_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Int2); 
        t.stop();
    });
    DspWorkspace<int16_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels, VecMode::Int2); 

    ProfileResult process_res = benchmark<GpuTimer>(measurementRounds, warmupRounds, [&](GpuTimer& t) {
        vload2AveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int16_t));

    logger.log(
        "Vectorized SM Parallel Averager",      
        "Standard",          
        samples.size(),      
        grade,              
        blockSize,           
        process_res,         
        sizeof(int16_t)      
    );

    cout << "\n--- MODE: UNIFIED (Zero-Copy) ---" << endl;

    ProfileResult init_res_uni = benchmark<CpuTimer>(measurementRounds, warmupRounds, [&](CpuTimer& t) {
        t.start();
        DspWorkspace<int16_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Int2); 
        t.stop();
    });
    DspWorkspace<int16_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels, VecMode::Int2); 

    ProfileResult process_res_uni = benchmark<GpuTimer>(measurementRounds, warmupRounds, [&](GpuTimer& t) {
        vload2AveragerGpuLoad(workspace_uni, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res_uni.initialization_ms = init_res_uni.compute_ms;
    process_res_uni.print_stats(samples.size(), sizeof(int16_t));

    logger.log(
        "Vectorized SM Parallel Averager",
        "Unified",          
        samples.size(),      
        grade,              
        blockSize,           
        process_res_uni,         
        sizeof(int16_t)      
    );
   
}

int32_t averager(const string pathName, const int blockSize, int point){
    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty())
        return -1; 
    uint32_t totalSamples = samples.size();
    vector<int16_t> processedSamples(totalSamples);
    
    vload2AveragerProfiler(header.numChannels, point, blockSize, samples, processedSamples);
    // disabling write for profiling
    // writeSamples("profile_sm_averager.wav" ,header, processedSamples);
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

