#include <iostream>
#include<fstream>
#include <vector>
#include <string>
#include <tuple>

#include "../wav_header.h"
#include "../benchmark.h"
#include "../gpu_utils.h"

using namespace std;


void profilable_cpu_computations(int numberOfChannels, int point, const vector<int16_t>& samples, vector<int16_t>& processedSamples){

    uint32_t totalSamples = samples.size() / numberOfChannels; // INP: this is total samples per channel
                                                                                     // NOT total samples in the file
    // processing first "point" samples of each channel                                                                                
    vector<int64_t> sampleSum(numberOfChannels, 0);

    for(int i = 0 ; i < point; i++){
        for(int ch = 0 ; ch < numberOfChannels; ch++){
            int16_t sample = samples[i*numberOfChannels + ch];
            sampleSum[ch] += sample;
            processedSamples[i*numberOfChannels + ch] = static_cast<int16_t>(sampleSum[ch] / point);
        }
    }

    for(uint32_t i = point ; i < totalSamples; i++){
        for(int ch = 0 ; ch < numberOfChannels; ch++){
            uint32_t currentIndex = i * numberOfChannels + ch;
            uint32_t trailingIndex = (i - point) * numberOfChannels + ch;
            sampleSum[ch] -= samples[trailingIndex];
            sampleSum[ch] += samples[currentIndex];
            processedSamples[i * numberOfChannels + ch] = static_cast<int16_t>(sampleSum[ch] / point);
        }
    }

}


// an N point moving averager
uint32_t averager(string pathName, int point){
    CsvLogger logger("benchmark_data.csv");

    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty()){
        return 1; // error in reading samples
    }

    uint32_t totalSamples = (header.dataBytes / (header.bitsPerSample / 8));
    cout<<"--- Single Thread Averager ---" << endl;
    cout<<"total samples: " << totalSamples << endl;
    cout<< "point: " << point << endl;

    ProfileResult init_res = benchmark<CpuTimer>(25, 5, [&](CpuTimer& t) {
        t.start();
        // Allocate and zero-init
        vector<int16_t> temp_buffer(samples.size()); 
        t.stop();
        // Vector is destroyed here (equivalent to free)
    });

    vector<int16_t> processedSamples(samples.size());

    ProfileResult process_res = benchmark<CpuTimer>(50, 10, [&](CpuTimer& t) {
        t.start();
        // Only measure the math
        profilable_cpu_computations(header.numChannels, point, samples, processedSamples);
        t.stop();
    });

    process_res.initialization_ms = init_res.compute_ms;

    process_res.print_stats(samples.size(), sizeof(int16_t));

    // Log to CSV
    logger.log(
        "SingleThreadCpu",   // Algorithm Name
        "RAM",               // Memory Mode
        samples.size(),      // N
        point,               // Grade (FIXED: Was 0)
        0,                   // Block Size (N/A for CPU)
        process_res,         // Results
        sizeof(int16_t)      // Input Size
    );
    
    writeSamples( "single_thread_averager.wav",header, processedSamples);
    return  totalSamples;// this is the actual total samples processed
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
    uint32_t result = averager(pathName, grade);

    return (result > 0) ? 0 : 1;
}
