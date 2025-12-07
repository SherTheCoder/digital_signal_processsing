#pragma once

#include <iostream>
#include <chrono>
#include <type_traits>
#include "wav_header.h"

template <typename Func>

void run_benchmark( 
                   int numRuns, 
                   WAVHeader header,
                   Func functionToBenchmark) 
{
    double totalDurationUs = 0.0;
    uint32_t numberOfSamples = header.dataBytes / (header.bitsPerSample / 8);
    
    using ResultType = decltype(functionToBenchmark());
    uint64_t totalSum = 0;

    std::cout << "Running benchmark: " << std::endl;
    std::cout << " Samples per run: " << numberOfSamples << endl;
    std::cout << " total runs:  " << numRuns << std::endl;

    for (int i = 0; i < numRuns; i++) {
        auto start = std::chrono::steady_clock::now();

        if constexpr (std::is_void_v<ResultType>) 
            functionToBenchmark();
        else 
            volatile ResultType result = functionToBenchmark();

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        totalDurationUs += duration.count();
    }

    double avgDurationUs = totalDurationUs / numRuns;
    double avgDurationMs = avgDurationUs / 1000.0;
    double avgDurationS = avgDurationMs / 1000.0;

    double throughputSamplesPerSec = 0.0;
    if (avgDurationS > 0) {
         throughputSamplesPerSec = numberOfSamples / avgDurationS;
    }

    std::cout << "--- Performance ---" << std::endl;
    std::cout << "Average Wall Clock Time: " << avgDurationMs << " ms (" << avgDurationUs << " µs)" << std::endl;
    std::cout << "Throughput:              " << (throughputSamplesPerSec / 1e6) << " Mega samples/sec" << std::endl<<std::endl;
    
    // Print the "used" result to ensure it's not optimized away
    std::cout << "Result check (to prevent optimization): " << totalSum << "\n" << std::endl;
}