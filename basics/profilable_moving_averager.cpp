#include <iostream>
#include<fstream>
#include <vector>
#include <string>
#include <tuple>

#include "../wav_header.h"
#include "../benchmark_chrono.h"

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
uint32_t singleThreadAverager(string pathName, int point){
    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty()){
        return 1; // error in reading samples
    }

    uint32_t totalSamples = (header.dataBytes / (header.bitsPerSample / 8));
    vector<int16_t> processedSamples(samples.size());
    cout<<"--- Single Thread Averager ---" << endl;
    cout<<"total samples: " << totalSamples << endl;
    cout<< "point: " << point << endl;
    run_benchmark(
        50, // Run this many times
        header,
        [&]() {
            profilable_cpu_computations(header.numChannels, point, samples, processedSamples);
        }
    );
    
    writeSamples( "single_thread_averager.wav",header, processedSamples);
    return  totalSamples;// this is the actual total samples processed
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
    cout<< "What should be the grade of the averager? ";
    cin>> point;
    if(point <= 0){
        cerr<< "Grade should be positive integer"<< endl;
        return 1;
    }
    uint32_t numberOfSamples = singleThreadAverager(pathName, point);

    if(numberOfSamples <= 0){
        cerr<< "something weird happened, code: " << numberOfSamples << endl;
        return 1;
    }
    
    return 0;
}

