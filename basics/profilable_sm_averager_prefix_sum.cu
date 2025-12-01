#include<iostream>
#include<fstream>
#include <vector>
#include <string>
#include <tuple>
#include<chrono>
#include<cuda_runtime.h>
#include "../wav_header.h"
#include "../benchmark.cpp"


using namespace std;


__global__
void uniform_add(const int total_samples, const int number_of_channels, const int64_t* __restrict__ aux, int64_t* samples){
    int bid = blockIdx.x;
    int dim = blockDim.x;
    int tid = threadIdx.x;
    uint32_t g_index = (uint32_t)bid * dim + tid;

    if(g_index >= total_samples) return;

    if(bid > 0){
        int channel = tid % number_of_channels;
        uint32_t aux_index = (uint32_t)(bid - 1) * number_of_channels + channel;
        samples[g_index] += aux[aux_index];
    }
        
}

// int64 for processed as it'll contain the whole sum [this'll NEVER overflow]
// int32 for sm, as the most number of samples it'll sum is the block size [i.e. 1024] [this'll NEVER overflow]
// UPDATE: changed samples to int64 as this will be called recursively and aux will contain large sums
__global__
void hillis_steele(const uint32_t total_elements, const int number_of_channels, const int64_t* __restrict__ samples, 
    int64_t* processedSamples, int64_t* aux_in_global_memory){
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


// N is the samples per channel, not total samples
void recursive_hillis_steele(const int block_size, const uint32_t total_samples, const int number_of_channels, int64_t*  samples){
    uint32_t number_of_blocks = (uint32_t)(total_samples + block_size - 1) / block_size;
    // base case
    if(number_of_blocks == 1){
        size_t size_of_shared_memory = total_samples * sizeof(int64_t);
        hillis_steele<<<1, total_samples, size_of_shared_memory>>>(total_samples, number_of_channels, samples, samples, NULL);
        return;
    }
    // DIVIDE
    int64_t* aux;
    cudaMalloc(&aux, number_of_blocks * number_of_channels * sizeof(int64_t));
    size_t size_of_shared_memory = block_size * sizeof(int64_t);
    // CONQUER
    // Phase 1: Scan this level and export sums to aux
    hillis_steele<<< number_of_blocks, block_size, size_of_shared_memory>>>(total_samples, number_of_channels, samples, samples, aux);
    // Phase 2: Recursion Scan the aux array!
    recursive_hillis_steele(block_size, number_of_blocks * number_of_channels, number_of_channels, aux);
    // COMBINE
    // Phase 3: Uniform Add
    uniform_add<<<number_of_blocks, block_size>>>(total_samples, number_of_channels, aux, samples);

    cudaFree(aux);
}


__global__
void averager_kernel(const int grade, const float inverseGrade, const int halo, const int numOfChannels, const int N, 
                            const int64_t* __restrict__ samples, int16_t* processedSamples){

    int blockSize = blockDim.x;
    int indexOfFirstThread = blockSize * blockIdx.x;
    uint32_t g_threadIndex = (uint32_t)indexOfFirstThread + threadIdx.x;
    extern __shared__ int64_t shared_memory[];

    int numberOfSamplesInSm = blockSize + halo;

    for(int i = threadIdx.x; i < numberOfSamplesInSm; i += blockSize){
        int global_idx = indexOfFirstThread + i;
        shared_memory[i] = samples[global_idx];
    }
    
    __syncthreads();

    if(g_threadIndex < N){
        int64_t sum = 0;
        int prev_index = halo + threadIdx.x - (grade * numOfChannels);
        sum = shared_memory[halo + threadIdx.x] - shared_memory[prev_index];
        processedSamples[g_threadIndex] = static_cast<int16_t>(sum * inverseGrade); // multiplication is faster than division
    }
}

vector<int16_t> profilable_moving_averager(const WAVHeader header, const vector<int16_t>& samples, int grade, int blockSize){
    int64_t *d_samples;
    int16_t *d_processedSamples;
    int totalSamples = samples.size();
    int halo = grade * header.numChannels;
    //move samples from host to device memory:
    //1. Allocate device memory
    // padding at the end with blockSize to simplify kernel logic
    cudaMalloc(&d_samples, (totalSamples + halo + blockSize) * sizeof(int64_t));
    cudaMemset(d_samples, 0, (totalSamples + halo + blockSize) * sizeof(int64_t)); 
    // 2. copy samples to device memory
    vector<int64_t> host_samples_64(samples.begin(), samples.end());
    cudaMemcpy(d_samples + halo, host_samples_64.data(), totalSamples * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    recursive_hillis_steele(blockSize, totalSamples, header.numChannels, d_samples + halo); // no need to run scan on first halo zeros

    // move complete
    cudaMalloc(&d_processedSamples, samples.size() * sizeof(int16_t));
    
    
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    int sharedMemorySize = (blockSize + halo) * sizeof(int64_t);
    float inverseGrade = 1.0f / static_cast<float>(grade);
    averager_kernel<<<gridSize, blockSize, sharedMemorySize>>>(grade, inverseGrade, halo, header.numChannels, totalSamples, d_samples, d_processedSamples);
    cudaDeviceSynchronize();

    //move samples from device to host memory
    vector<int16_t> processedSamples(totalSamples);
    cudaMemcpy(processedSamples.data(), d_processedSamples, totalSamples * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_samples);
    cudaFree(d_processedSamples);
    return processedSamples;
}

int32_t averager(const string pathName, const int blockSize, int point){
    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty())
        return -1; // error in reading samples
    vector<int16_t> processedSamples;
    uint32_t totalSamples = samples.size();
    
    run_benchmark(
        200, // Run this many times
        header,
        [&]() {
            processedSamples = profilable_moving_averager(header, samples, point, blockSize);
        }
    );

    writeSamples("profile_sm_averager.wav" ,header, processedSamples);
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

