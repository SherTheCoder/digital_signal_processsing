#include<iostream>
#include<fstream>
#include <vector>
#include <string>
#include <tuple>
#include<chrono>
#include<cuda_runtime.h>
#include "../wav_header.h"

using namespace std;

__global__
void averager_kernel(const int grade, const int numOfChannels, const int N, const int16_t* samples, int16_t* processedSamples){
    int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    // to minimize accesses to unified memory, we'll use a local sum variable
    if(threadIndex < N){
        int32_t sum = 0;
        // let's unroll the loop as it is serial
        #pragma unroll
        for(int i = 0 ; i < grade; i++)
            sum += samples[numOfChannels * grade + threadIndex - i * numOfChannels];
        processedSamples[threadIndex] = static_cast<int16_t>(sum / grade);
    }
}
// we're assuming the host does not have unified memory. 
// If it does, there's no need to copy to device and back.
vector<int16_t> profilable_moving_averager(const WAVHeader header, const vector<int16_t>& samples, int grade, int blockSize){
    int16_t *d_samples;
    int16_t *d_processedSamples;
    int totalSamples = samples.size();
    //move samples from host to device memory:
    //1. Allocate device memory
    cudaMalloc(&d_samples, (totalSamples + grade * header.numChannels) * sizeof(int16_t));
    // 2. copy samples to device memory
    cudaMemcpy(d_samples + grade*header.numChannels, samples.data(), totalSamples * sizeof(int16_t), cudaMemcpyHostToDevice);
    // move complete
    cudaMalloc(&d_processedSamples, samples.size() * sizeof(int16_t));

    //set first grade samples for each channel to zero
    cudaMemset(d_samples, 0, grade * header.numChannels * sizeof(int16_t)); 
    
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    averager_kernel<<<gridSize, blockSize>>>(grade, header.numChannels, totalSamples, d_samples, d_processedSamples);
    cudaDeviceSynchronize();

    //move samples from device to host memory
    vector<int16_t> processedSamples(totalSamples);
    cudaMemcpy(processedSamples.data(), d_processedSamples, totalSamples * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_samples);
    cudaFree(d_processedSamples);
    return processedSamples;
}

int averager(string pathName, int point, int blockSize){
    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty())
        return -1; // error in reading samples
    vector<int16_t> processedSamples;
    uint32_t totalSamples = samples.size();

    run_benchmark(
        100, // Run this many times
        header,
        [&]() {
            processedSamples = profilable_moving_averager(header, samples, point, blockSize);
        }
    );

    writeSamples(header, processedSamples);
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
    if(blockSize <= 32 || blockSize >= 1024 || blockSize%32 != 0){
        cout<<"blockSize should be multiple of 32, >=32 && <=1024"<< endl;
        return -1;
    }
    // To keep an apples to apples comparison, the function handles everything once the inputs are given. 
    uint32_t numOfSamples = averager(pathName, point, blockSize);
    if(numOfSamples <= 0)
        cout<< "something weird happened, code: " << numOfSamples << endl;

    return 0;
}

