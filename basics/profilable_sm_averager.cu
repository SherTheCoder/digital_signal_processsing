#include<iostream>
#include<fstream>
#include <vector>
#include <string>
#include <tuple>
#include<cuda_runtime.h>
#include "../wav_header.h"
#include "../benchmark.h"
#include "../gpu_utils.h"

// grade - 1 is the number of duplicate calls we make to the VRAM per block
// the larger the grade, the higher duplicate calls

// the larger the block size, the fewer the duplicate calls, but higher occupancy issues

// Nice problem : find the optimal block size given a grade
// for small grade, a small block size is good
// for large grade, we might get to a point where the duplicate calls per block to VRAM exceed the 
//      occupancy advantage of a small block

using namespace std;

__global__
void averager_kernel(const int grade, const double inverseGrade, const int halo, const int numOfChannels, const int N, 
                            const int16_t* __restrict__ samples, int16_t* __restrict__ processedSamples){

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int block_start_idx = blockIdx.x * blockSize;
    const int g_threadIndex = block_start_idx + tid;

    extern __shared__ int16_t shared_memory[];

    const int numberOfSamplesInSm = blockSize + halo;

    for(int i = tid; i < numberOfSamplesInSm; i += blockSize){
        shared_memory[i] = samples[block_start_idx - halo + i];
    }
    
    __syncthreads();

    if(g_threadIndex < N){
        int64_t sum = 0;
        // pointer to this thread's current sample in shared memory
        const int16_t* currentWindow = &shared_memory[halo + tid];
        #pragma unroll
        for(int i = 0 ; i < grade; i++)
            sum += currentWindow[- (i * numOfChannels)];
        processedSamples[g_threadIndex] = static_cast<int16_t>(sum * inverseGrade); // multiplication is faster than division
    }
}

void sharedMemoryAveragerGpuLoad(const DspWorkspace<int16_t>& workspace, const int grade, const int blockSize, const int numOfChannels, GpuTimer& t, 
                                                const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    t.start();
    int16_t *d_samples = workspace.input_valid;
    int16_t *d_processedSamples = workspace.output;

    int totalSamples = samples.size();
    int halo = static_cast<int>(workspace.halo_elements);

    // copy samples to device memory
    CUDA_CHECK(
        cudaMemcpy(d_samples, samples.data(), totalSamples * sizeof(int16_t), cudaMemcpyHostToDevice)
    );
    // move complete
    t.mark_h2d();
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    int sharedMemorySize = (blockSize + halo) * sizeof(int16_t);
    double inverseGrade = 1.0f / static_cast<double>(grade);
    averager_kernel<<<gridSize, blockSize, sharedMemorySize>>>(grade, inverseGrade, halo, numOfChannels, totalSamples, d_samples, d_processedSamples);
    t.mark_compute();
    //move samples from device to host memory
    CUDA_CHECK(
        cudaMemcpy(processedSamples.data(), d_processedSamples, totalSamples * sizeof(int16_t), cudaMemcpyDeviceToHost)
    );
    t.stop();
}

void sharedMemoryAveragerProfiler(const int numOfChannels, const int grade, const int blockSize, 
                                                const vector<int16_t>& samples, vector<int16_t>& processedSamples){
    // CPU benchmarking (cudaMalloc and cudaFree)
    ProfileResult init_res = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        // Constructor runs cudaMalloc
        DspWorkspace<int16_t> workspace(samples.size(), grade, numOfChannels); 
        t.stop();
        // Destructor runs cudaFree AUTOMATICALLY here (end of scope)
    });

    DspWorkspace<int16_t> workspace(samples.size(), grade, numOfChannels);

    // GPU benchmarking (cudaMemcpy from host -> kernel execution -> cudaMemcpy to host)
    ProfileResult process_res = benchmark<GpuTimer>(50, 10, [&](GpuTimer& t) {
        // Pass the workspace object
        sharedMemoryAveragerGpuLoad(workspace, grade, blockSize, numOfChannels, t, samples, processedSamples);
    });

    process_res.initialization_ms = init_res.compute_ms;
    process_res.print_stats(samples.size(), sizeof(int16_t));
}

int32_t averager(const string pathName, const int blockSize, int point){
    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty())
        return -1; // error in reading samples
    uint32_t totalSamples = samples.size();
    vector<int16_t> processedSamples(totalSamples);

    cout<<"--- SHARED MEMORY PARALLEL AVERAGER ---"<< endl;
    cout<<"Samples: "<< samples.size() << endl;
    cout<<"point: " << point << endl;
    cout<<"block Size: "<< blockSize<< endl;
    
    sharedMemoryAveragerProfiler(header.numChannels, point, blockSize, samples, processedSamples);
    
    writeSamples("shared_memory_averager.wav", header, processedSamples);
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

