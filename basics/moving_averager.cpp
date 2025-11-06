#include <iostream>
#include<fstream>
#include "../wav_header.h"
#include <chrono>    // For timing
using namespace std;

// an N point moving averager
int averager(string pathName, int point){
    // opening the file
    ifstream input(pathName, ios::binary);
    if(!input)
        return -100;
    WAVHeader header;
    // read actually needs a character pointer, and the size of bytes to read
    // reinterpret cast tells C++ to treat header as a character pointer, and trust it will all go well
    input.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    if(header.bitsPerSample == 64 || header.bitsPerSample == 8 || header.bitsPerSample == 24 || header.bitsPerSample == 32)
        return -10; // it will overflow as the largest built in int we have is int64_t
                    // Also, 8 bit and 24 bit are currently unsupported as they require
                    // a piece of code to "sign extend" the int64_t variable we're using to read from input
    

    ofstream output("output.wav", ios::out | ios::binary);
    if(!output)
        return -20;
    output.write(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    // This will be a circular buffer to hold the previous N samples
    // and let us calculate a running average without having a second pointer in our input file
    int numberOfChannels = header.numChannels;
    vector<vector<int16_t>> prevSamples(numberOfChannels, vector<int16_t>(point, 0));
    uint32_t prevSamplesIndex = 0;
    // This will not be an unsigned as the sign matters when adding samples  
    vector<int32_t> sampleSum(numberOfChannels, 0);
    int bytesPerSample = header.bitsPerSample / 8;
    // DEBUG printing:
    // cout << "sizeof(WAVHeader) = " << sizeof(WAVHeader) << "\n";
    // cout << "riff = " << string(header.riff, 4) << "\n";
    // cout << "wave = " << string(header.wave, 4) << "\n";
    // cout << "fmtSize = " << header.fmtSize << "\n";
    // cout << "bitsPerSample = " << header.bitsPerSample << "\n";
    // cout << "data tag = " << string(header.data, 4) << "\n";
    // cout << "dataBytes = " << header.dataBytes << "\n";
    // cout<< bytesPerSample<<endl;
    // cout<< numberOfChannels<<endl;
    
    uint32_t totalSamples = header.dataBytes / (bytesPerSample * numberOfChannels);
    

    for(uint32_t i = 0 ; i < totalSamples; i++){
        for(int ch = 0 ; ch < numberOfChannels; ch++){
            int16_t sample = 0;
            input.read(reinterpret_cast<char*>(&sample), bytesPerSample);
            sampleSum[ch] -= prevSamples[ch][prevSamplesIndex % point];
            sampleSum[ch] += sample;
            prevSamples[ch][prevSamplesIndex % point] = sample;
            int16_t avg = static_cast<int16_t> (sampleSum[ch] / point);
            output.write(reinterpret_cast<char*>(&avg), bytesPerSample);
        }
        prevSamplesIndex++;
    }

    input.close();
    output.close();
    return totalSamples;

    
}

int main(){
    
    string pathName;
    int point;
    cout<< "Enter the path of the .wav file: ";
    cin>> pathName;
    cout<< "What should be the grade of the averager? ";
    cin>> point;
    if(point <= 0)
        cout<< "don't play";
    uint32_t numberOfSamples = averager(pathName, point);
    if(numberOfSamples <= 0)
        cout<< "something weird happened, code: " << numberOfSamples << endl;



    // For benchmarking
    int numRuns = 10; // Average over multiple runs for stability
    double totalDurationUs = 0.0;
    uint64_t totalSum = 0.0; // To prevent compiler from optimizing away
    uint32_t result;
    cout<< "running the benchmark with chrono. Total samples: "<< numberOfSamples << endl;
    for(int i = 0 ; i < numRuns; i++){
        auto start = chrono::steady_clock::now();
        result = averager(pathName, point);
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

    double throughputSamplesPerSec = numberOfSamples / avgDurationS;
    cout << "--- CPU Performance ---"<<endl;
    cout << "Average Wall Clock Time: " << avgDurationMs << " ms (" << avgDurationUs << " µs)"<<endl;
    cout << "Throughput:              " << (throughputSamplesPerSec / 1e6) << " Mega samples/sec"<<endl;
    cout<< "Total samples processed:  " << numberOfSamples << endl;

    
    // Print the "used" result to ensure it's not optimized away
    cout << "Result check (to prevent optimization): " << totalSum << "\n";

    
    return 0;
}