#include<iostream>
#include<fstream>
#include <vector>
#include <string>
#include <tuple>
#include<cuda_runtime.h>
#include "../wav_header.h"
#include "../benchmark.h"
#include "../gpu_utils.h"

using namespace std;

__global__
void averager_kernel(const int grade, const int numOfChannels, const int N, const int16_t* __restrict__ samples, int16_t* __restrict__ processedSamples){
    int gIndex = blockDim.x * blockIdx.x + threadIdx.x;
    // to minimize accesses to unified memory, we'll use a local sum variable
    if(gIndex < N){
        int64_t sum = 0;
        // let's unroll the loop as it is serial
        #pragma unroll
        for(int i = 0 ; i < grade; i++)
            sum += samples[gIndex - i * numOfChannels];
        processedSamples[gIndex] = static_cast<int16_t>(sum / grade);
    }
}
// we're assuming the host does not have unified memory. 
// If it does, there's no need to copy to device and back.
template <typename T, MemoryMode Mode>
void parallelAveragerGpuLoad(const DspWorkspace<T, Mode>& workspace, const int grade, const int blockSize, const int numOfChannels, GpuTimer& t, 
                                                const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    t.start();
    int16_t *d_samples = workspace.input_valid;
    int16_t *d_processedSamples = workspace.output;

    int totalSamples = samples.size();

    // copy samples to device memory
    MemoryTraits<Mode>::copyH2D(
        d_samples, 
        samples.data(), 
        totalSamples * sizeof(T)
    );

    // move complete
    t.mark_h2d();
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    averager_kernel<<<gridSize, blockSize>>>(grade, numOfChannels, totalSamples, d_samples, d_processedSamples);
    t.mark_compute();
    //move samples from device to host memory
    MemoryTraits<Mode>::copyD2H(
        processedSamples.data(), 
        d_processedSamples, 
        totalSamples * sizeof(int16_t)
    );

    t.stop();
}

void simpleParallelAveragerProfiler(const int numOfChannels, const int grade, const int blockSize, 
                                                const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    CsvLogger logger("benchmark_data.csv");

    cout << "\n--- MEM MODE: STANDARD (Discrete) ---" << endl;
    // CPU benchmarking (cudaMalloc and cudaFree)
    ProfileResult init_res = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        // Constructor runs cudaMalloc
        DspWorkspace<int16_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels); 
        t.stop();
        // Destructor runs cudaFree AUTOMATICALLY here (end of scope)
    });

    DspWorkspace<int16_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels);

    // GPU benchmarking (cudaMemcpy from host + kernel execution + cudaMemcpy to host)
    ProfileResult process_res = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace object
        parallelAveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });
    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int16_t));

    logger.log(
        "Parallel Averager",      // Algorithm Name
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
        DspWorkspace<int16_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels); 
        t.stop();
        // Destructor runs cudaFree AUTOMATICALLY here (end of scope)
    });

    DspWorkspace<int16_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels);

    // GPU benchmarking (cudaMemcpy from host + kernel execution + cudaMemcpy to host)
    ProfileResult process_res_uni = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace object
        parallelAveragerGpuLoad(workspace_uni, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });
    process_res_uni.initialization_ms = init_res_uni.compute_ms;
    process_res_uni.print_stats(samples.size(), sizeof(int16_t));

    logger.log(
        "Parallel Averager",      // Algorithm Name
        "Unified",         
        samples.size(),      
        grade,               
        blockSize,           
        process_res_uni,         
        sizeof(int16_t)      
    );

}

uint32_t averager(string pathName, int blockSize, int point){
    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty())
        return 1; // error in reading samples
    vector<int16_t> processedSamples(samples.size());
    uint32_t totalSamples = samples.size();

    cout<<"--- SIMPLE PARALLEL AVERAGER ---"<< endl;
    cout<<"Samples: "<< samples.size() << endl;
    cout<<"point: " << point << endl;
    cout<<"block Size: "<< blockSize<< endl;

    simpleParallelAveragerProfiler(header.numChannels, point, blockSize, samples, processedSamples);

    // writeSamples("profilable_parallel_averager.wav",header, processedSamples);
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

