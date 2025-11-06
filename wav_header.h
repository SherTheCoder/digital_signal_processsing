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

