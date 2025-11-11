#include <iostream>
#include<fstream>
#include "../wav_header.h"
#include "../benchmark.cpp"
using namespace std;


vector<int16_t> profilable_cpu_computations(int numberOfChannels, const vector<int16_t>& samples, int point){
    vector<int16_t> processedSamples(samples.size());

    uint32_t totalSamples = samples.size() / numberOfChannels; // INP: this is total samples per channel
                                                                                     // NOT total samples in the file
    // processing first "point" samples of each channel                                                                                
    vector<int32_t> sampleSum(numberOfChannels, 0);
    for(int i = 0 ; i < point; i++){
        for(int ch = 0 ; ch < numberOfChannels; ch++){
            int16_t sample = samples[i*numberOfChannels + ch];
            sampleSum[ch] += sample;
            processedSamples[i*numberOfChannels + ch] = static_cast<int16_t>(sampleSum[ch] / point);
        }
    }

    for(uint32_t i = point ; i < totalSamples; i++){
        for(int ch = 0 ; ch < numberOfChannels; ch++){
            int16_t sample = samples[i * numberOfChannels + ch];
            sampleSum[ch] -= samples[(i - point)*numberOfChannels + ch];
            sampleSum[ch] += sample;
            processedSamples[i * numberOfChannels + ch] = static_cast<int16_t>(sampleSum[ch] / point);
        }
    }

    return processedSamples;
}


// an N point moving averager
int averager(string pathName, int point){
    vector<int16_t> samples;
    WAVHeader header;
    tie(header, samples) = extractSamples(pathName);
    if(samples.empty()){
        return -1; // error in reading samples
    }
    uint32_t totalSamples = (header.dataBytes / (header.bitsPerSample / 8));
    vector<int16_t> processedSamples;

    run_benchmark(
        1000, // Run this many times
        header,
        [&]() {
            processedSamples = profilable_cpu_computations(header.numChannels, samples, point);
        }
    );
    
    writeSamples(header, processedSamples);
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
    if(point <= 0)
        cout<< "don't play";
    uint32_t numberOfSamples = averager(pathName, point);
    if(numberOfSamples <= 0)
        cout<< "something weird happened, code: " << numberOfSamples << endl;
    
    return 0;
}





//RAM USAGE: 492KB peak for 12 point averager