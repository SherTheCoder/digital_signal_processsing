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
void averager_kernel(const int grade, const float inverseGrade, const int halo, const int numOfChannels, const int N, 
                            const int16_t* __restrict__ samples, int16_t* __restrict__ processedSamples){

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int block_start_idx = blockIdx.x * blockSize;
    const int g_threadIndex = block_start_idx + tid;

    extern __shared__ int16_t shared_memory[];

    const int numberOfSamplesInSm = blockSize + halo;

    for(int i = tid; i < numberOfSamplesInSm; i += blockSize){
        int index = block_start_idx - halo + i;
        if(index < N)
        shared_memory[i] = samples[block_start_idx - halo + i];
        else
        shared_memory[i] = 0;
    }
    
    __syncthreads();

    if(g_threadIndex < N){
        int64_t sum = 0;
        const int16_t* currentWindow = &shared_memory[halo + tid];
        #pragma unroll
        for(int i = 0 ; i < grade; i++)
            sum += currentWindow[- (i * numOfChannels)];
        processedSamples[g_threadIndex] = static_cast<int16_t>(sum * inverseGrade);
    }
}

template <typename T, MemoryMode Mode>
void sharedMemoryAveragerGpuLoad(const DspWorkspace<T, Mode>& workspace, const int grade, const int blockSize, const int numOfChannels, GpuTimer& t, 
                                                const vector<int16_t>& samples, vector<int16_t>& processedSamples){
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
    int sharedMemorySize = (blockSize + halo) * sizeof(int16_t);
    float inverseGrade = 1.0f / static_cast<float>(grade);
    averager_kernel<<<gridSize, blockSize, sharedMemorySize>>>(grade, inverseGrade, halo, numOfChannels, totalSamples, d_samples, d_processedSamples);
    t.mark_compute();
    MemoryTraits<Mode>::copyD2H(
        processedSamples.data(), 
        d_processedSamples, 
        totalSamples * sizeof(int16_t)
    );
    t.stop();
}

void sharedMemoryAveragerProfiler(const int numOfChannels, const int grade, const int blockSize, 
                                                const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    CsvLogger logger("benchmark_data.csv");
    cout << "\n--- MEM MODE: STANDARD (Discrete) ---" << endl;
    ProfileResult init_res = benchmark<CpuTimer>(measurementRounds, warmupRounds, [&](CpuTimer& t) {
        t.start();
        DspWorkspace<int16_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels); 
        t.stop();
    });

    DspWorkspace<int16_t, MemoryMode::Standard> workspace(samples.size(), grade, numOfChannels);

    ProfileResult process_res = benchmark<GpuTimer>(measurementRounds, warmupRounds, [&](GpuTimer& t) {
        sharedMemoryAveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int16_t));

    logger.log(
        "SM Parallel Averager",
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
        DspWorkspace<int16_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels);
        t.stop();
    });

    DspWorkspace<int16_t, MemoryMode::Unified> workspace_uni(samples.size(), grade, numOfChannels);

    ProfileResult process_res_uni = benchmark<GpuTimer>(measurementRounds, warmupRounds, [&](GpuTimer& t) {
        sharedMemoryAveragerGpuLoad(workspace_uni, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res_uni.initialization_ms = init_res_uni.compute_ms;
    process_res_uni.print_stats(samples.size(), sizeof(int16_t));

    logger.log(
        "SM Parallel Averager",
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

    cout<<"--- SHARED MEMORY PARALLEL AVERAGER ---"<< endl;
    cout<<"Samples: "<< samples.size() << endl;
    cout<<"point: " << point << endl;
    cout<<"block Size: "<< blockSize<< endl;
    
    sharedMemoryAveragerProfiler(header.numChannels, point, blockSize, samples, processedSamples);
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

