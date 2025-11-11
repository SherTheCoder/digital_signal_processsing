#include <iostream>
#include<fstream>
#include "../wav_header.h"
#include "../benchmark.cpp"
using namespace std;


pair<WAVHeader, vector<int16_t>> extractSamples(string pathName) {
    ifstream input(pathName, ios::binary);
    if(!input){
        std::cout<< "could not open file"<<endl;
        return {};
    }
    WAVHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    if(header.bitsPerSample == 64 || header.bitsPerSample == 8 || header.bitsPerSample == 24 || header.bitsPerSample == 32){
        std:: cout<< "unsupported bits per sample: " << header.bitsPerSample << endl;
        return {};
    }
    int bytesPerSample = header.bitsPerSample / 8;
    int numOfSamples = header.dataBytes / bytesPerSample;
    vector<int16_t> samples(numOfSamples);
    for (int i = 0; i < numOfSamples; ++i) {
        int16_t sample = 0;
        input.read(reinterpret_cast<char*>(&sample), bytesPerSample);
        samples[i] = sample;
    }
    input.close();
    return {header, samples};
}

void writeSamples(const WAVHeader header, vector<int16_t>& samples) {
    ofstream output("output_averaged.wav", ios::out | ios::binary);
    if(!output){
        std::cout<< "could not open output file"<<endl;
        return;
    }
    output.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));
    output.write(reinterpret_cast<const char*>(samples.data()), samples.size() * sizeof(int16_t));
    output.close();
}

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
    uint32_t totalSamples = (header.dataBytes / (header.bitsPerSample / 8));
    if(samples.empty()){
        return -1; // error in reading samples
    }
    vector<int16_t> processedSamples;

    run_benchmark(
        1000, // Run this one more times
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