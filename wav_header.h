#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
using namespace std;

#pragma pack(push, 1) // Ensure no padding is added
struct WAVHeader {
    char riff[4];        // "RIFF"
    uint32_t sizeOfFile;  // Size of the file minus 8 bytes
    char wave[4];       // "WAVE"
    char fmt[4];        // "fmt "
    uint32_t fmtSize; // Size of the fmt chunk
    uint16_t audioFormat; // Audio format (1 for PCM)
    uint16_t numChannels;  // Number of channels
    uint32_t sampleRate;      // Sample rate
    uint32_t byteRate;        // Byte rate
    uint16_t blockAlign;    // Block align
    uint16_t bitsPerSample;  // Bits per sample
    char data[4]; // "data"
    uint32_t dataBytes; // Size of the data chunk in bytes
};
#pragma pack(pop)

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

