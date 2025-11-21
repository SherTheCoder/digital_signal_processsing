#include<iostream>
#include<fstream>
#include <vector>
#include <string>
#include <tuple>
#include<chrono>
#include<cuda_runtime.h>
#include "../wav_header.h"
#include "../benchmark.cpp"

// grade - 1 is the number of duplicate calls we make to the VRAM per block
// the larger the grade, the higher the duplicate calls

// the larger the block size, the fewer the duplicate calls, but higher occupancy issues

// Nice problem : find the optimal block size given a grade
// for small grade, a small block size is good
// for large grade, we might get to a point where the duplicate calls per block to VRAM exceed the 
//      occupancy advantage of a small block

using namespace std;

//__restrict__ means read only, helps compiler optimize memory accesses
__global__
void averager_kernel(const float inverseGrade, const int grade, const int numOfChannels, const int N, 
    const int16_t* __restrict__ samples, int16_t* processedSamples, const int halo){
    extern __shared__ int16_t shared_memory[];
    const int2* vec_samples = reinterpret_cast<const int2*>(samples);
    int blockSize = blockDim.x;
    // How many int16_t elements we need in total
    int total_16bit_samples_needed = blockSize + halo;
    int total_vectors_needed = (total_16bit_samples_needed + 3) / 4; // each int2 has 4 int16_t. Rounding off. 

    int indexOfFirstThread = blockSize * blockIdx.x;
    for(int i = threadIdx.x; i < total_vectors_needed; i += blockSize){
        int global_vec_idx = (indexOfFirstThread / 4) + i;
        if (global_vec_idx * 4 < N + halo) { 
            // A. THE LOAD (1 Instruction, 4 Samples)
            int2 vector_data = vec_samples[global_vec_idx]; 
            // B. THE UNPACK
            // We need to extract 4 shorts and place them into shared_memory
            // Target index in shared memory
            int sm_base_idx = i * 4;
            // -- Unpack vector_data.x (Samples 0 and 1) --
            shared_memory[sm_base_idx]     = (int16_t)(vector_data.x & 0xFFFF);       // Bottom 16 bits
            shared_memory[sm_base_idx + 1] = (int16_t)((vector_data.x >> 16) & 0xFFFF); // Top 16 bits
            // -- Unpack vector_data.y (Samples 2 and 3) --
            shared_memory[sm_base_idx + 2] = (int16_t)(vector_data.y & 0xFFFF);       // Bottom 16 bits
            shared_memory[sm_base_idx + 3] = (int16_t)((vector_data.y >> 16) & 0xFFFF); // Top 16 bits
        }
    }
    
    int g_threadIndex = indexOfFirstThread + threadIdx.x;
    int cut = min(indexOfFirstThread + blockSize + halo, N + (grade - 1)*numOfChannels);
    
    __syncthreads();

    if(g_threadIndex < N){
        int32_t sum = 0;
        #pragma unroll
        for(int i = 0 ; i < grade; i++)
            sum += shared_memory[halo + threadIdx.x - i * numOfChannels];
        processedSamples[g_threadIndex] = static_cast<int16_t>(sum * inverseGrade); // multiply by inverseGrade instead of division, as multiplication is faster
    }
}

vector<int16_t> profilable_moving_averager(const WAVHeader header, const vector<int16_t>& samples, int grade, int blockSize){
    int16_t *d_samples;
    int16_t *d_processedSamples;
    int totalSamples = samples.size();
    int halo = (grade - 1) * header.numChannels;
    // as we're using vectorized loading, with in2 (4 int16 samples in one go)
    // we need number of samples to be multiple of 4
    int paddedSamplesAtEnd = 4 - (totalSamples + halo) % 4;
    if(paddedSamplesAtEnd == 4)
        paddedSamplesAtEnd = 0;
    //move samples from host to device memory:
    //1. Allocate device memory
    cudaMalloc(&d_samples, (totalSamples + halo + paddedSamplesAtEnd) * sizeof(int16_t));
    // 2. copy samples to device memory
    cudaMemset(d_samples, 0, (totalSamples + halo + paddedSamplesAtEnd) * sizeof(int16_t));
    cudaMemcpy(d_samples + halo, samples.data(), totalSamples * sizeof(int16_t), cudaMemcpyHostToDevice);
    // move complete
    cudaMalloc(&d_processedSamples, samples.size() * sizeof(int16_t));

    //now the first halo and the last paddedSamplesAtEnd samples are zeroes
    
    
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
    if(numberOfSamplesInSharedMemory % 4 != 0)
        numberOfSamplesInSharedMemory += 4 - (numberOfSamplesInSharedMemory % 4);
    sharedMemorySize = numberOfSamplesInSharedMemory * sizeof(int16_t);
    float inverseGrade = 1.0f / static_cast<float>(grade);
    averager_kernel<<<gridSize, blockSize, sharedMemorySize>>>(inverseGrade, grade, header.numChannels, totalSamples, d_samples, d_processedSamples, halo);
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

