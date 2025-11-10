#include<iostream>
#include<fstream>
#include "../wav_header.h"
#include<chrono>
#include<cuda_runtime.h>

// grade - 1 is the number of duplicate calls we make to the VRAM per block
// the larger the grade, the higher the duplicate calls

// the larger the block size, the fewer the duplicate calls, but higher occupancy issues

// Nice problem : find the optimal block size given a grade
// for small grade, a small block size is good
// for large grade, we might get to a point where the duplicate calls per block to VRAM exceed the 
//      occupancy advantage of a small block

using namespace std;

__global__
void averager_kernel(const int grade, const int numOfChannels, const int N, const int16_t* samples, int16_t* processedSamples){
    int blockSize = blockDim.x;
    int blockIndex = blockSize * blockIdx.x;
    int threadIndex = blockIndex + threadIdx.x;
    extern __shared__ int16_t shared_memory[];
    // the size of this shared memory is block size + (number of channels * (grade - 1))
    // now we want to allocate the work of populating this sm to threads in this block
    int tileSize = blockSize + (grade - 1)* numOfChannels;
    for(uint32_t i = threadIndex; i < N + grade*numOfChannels && i < blockIndex + tileSize; i += blockSize)
        shared_memory[i - blockIndex] = samples[i];

    __syncthreads();

    if(threadIndex < N){
        int32_t sum = 0;
        for(int i = 0 ; i < grade; i++)
            sum += (shared_memory[numOfChannels * (grade - 1) + threadIndex - blockIndex - i * numOfChannels]);
        processedSamples[threadIndex] = static_cast<int16_t>(sum / grade);
    }
}

int32_t averager(const string pathname, const int blockSize, int point){
    ifstream input(pathname, ios::binary);
    if(!input)
        return -1;
    WAVHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    // focus on 16 bit depth for simplicity
    if(header.bitsPerSample == 64 || header.bitsPerSample == 8 || header.bitsPerSample == 24 || header.bitsPerSample == 32)
        return -10;
    int numOfChannels = header.numChannels;
    int bytesPerSample = header.bitsPerSample / 8;
    uint32_t numOfSamplesPerChannel = header.dataBytes / (bytesPerSample * numOfChannels);


    // Let't create CUDA unified memory to hold all samples in a flat 1D vector
    int16_t* d_samples;
    int16_t* d_averagedSamples;
    uint32_t totalSamples = numOfSamplesPerChannel * numOfChannels;
    // let's add the first "point" samples and initialize them to zero to make the algorithm of threads cleaner
    cudaMallocManaged(&d_samples, totalSamples * sizeof(int16_t) + point * numOfChannels * sizeof(int16_t));
    cudaMallocManaged(&d_averagedSamples, totalSamples * sizeof(int16_t));
    // initialize the first "point" samples to zero
    for(int i = 0 ; i < numOfChannels * point; i++)
        d_samples[i] = 0;
    // There'll be garbage values in the first "point" samples of d_averagedSamples

    int16_t sample;
    // we'll store the samples interleaved by channel, like in original WAV file
    // This will help in correct averaging
    for(uint32_t i = 0 ; i < numOfSamplesPerChannel; i++){
        for(int ch = 0; ch < numOfChannels; ch++){
            input.read(reinterpret_cast<char*>(&sample), bytesPerSample);
            d_samples[ numOfChannels * point + (ch + i * numOfChannels)] = sample;
        }
    }

    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    sharedMemorySize = (blockSize + numOfChannels*(point - 1)) * sizeof(int16_t);
    averager_kernel<<<gridSize, blockSize, sharedMemorySize>>>(point, numOfChannels, totalSamples, d_samples, d_averagedSamples);
    cudaDeviceSynchronize();

    ofstream output("output_cuda", ios::out | ios::binary);
    if(!output)
        return -2;
    output.write(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    output.write(reinterpret_cast<char*>(d_averagedSamples), totalSamples * sizeof(int16_t));
    

    input.close();
    output.close();
    cudaFree(d_samples);
    cudaFree(d_averagedSamples);

    return totalSamples;
}

int main(){
    string pathname;
    cout<< "Enter pathname of wav file: ";
    cin>> pathname;
    int point;
    cout<< "Enter grade for moving averager: ";
    cin>> point;
    cout<< "Block Size: (should be multiple of 32)";
    int blockSize;
    cin>>blockSize;
    if(blockSize < 32 || blockSuze > 1024 || blockSize % 32 != 0){
        cout<< "don't play, bs should be multiple of 32, >= 32, <=1024"<< endl;
        return -150;
    }

    // To keep an apples to apples comparison, the function handles everything once the inputs are given. 
    int32_t numOfSamples = averager(pathname, blockSize, point);

    if(numOfSamples <= 0)
        cout<< "something weird happened, code: " << numOfSamples << endl;
    

    // For benchmarking
    int numRuns = 10; // Average over multiple runs for stability
    double totalDurationUs = 0.0;
    uint64_t totalSum = 0.0; // To prevent compiler from optimizing away
    uint32_t result;
    cout<< "running the benchmark with chrono. Total samples: "<< numOfSamples << endl;
    for(int i = 0 ; i < numRuns; i++){
        auto start = chrono::steady_clock::now();
        result = averager(pathname, blockSize, point);
        auto end = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        totalDurationUs += duration.count();
        
        // "Use" the result to prevent compiler optimization
        // A simple sum should do it.
        totalSum += result / 100;
    }

    double avgDurationUs = totalDurationUs / numRuns;
    double avgDurationMs = avgDurationUs / 1000.0;
    double avgDurationS = avgDurationMs / 1000.0;

    double throughputSamplesPerSec = numOfSamples / avgDurationS;
    cout << "--- CPU Performance ---"<<endl;
    cout << "Average Wall Clock Time: " << avgDurationMs << " ms (" << avgDurationUs << " µs)"<<endl;
    cout << "Throughput:              " << (throughputSamplesPerSec / 1e6) << " Mega samples/sec"<<endl;
    cout<< "Total samples processed:  " << numOfSamples << endl;

    
    // Print the "used" result to ensure it's not optimized away
    cout << "Result check (to prevent optimization): " << totalSum << "\n";

    return 0;
}

