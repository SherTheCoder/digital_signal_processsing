#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>

struct ProfileResult {
    float initialization_ms = 0.0f; 
    float transfer_h2d_ms = 0.0f;
    float compute_ms = 0.0f;
    float transfer_d2h_ms = 0.0f;
    float total_ms = 0.0f;

    void operator+=(const ProfileResult& other) {
        initialization_ms += other.initialization_ms;
        transfer_h2d_ms += other.transfer_h2d_ms;
        compute_ms      += other.compute_ms;
        transfer_d2h_ms += other.transfer_d2h_ms;
        total_ms        += other.total_ms;
    }

    void divide(int k) {
        if (k == 0) return;
        initialization_ms /= k;
        transfer_h2d_ms /= k;
        compute_ms      /= k;
        transfer_d2h_ms /= k;
        total_ms        /= k;
    }

    void print_stats(size_t num_elements, size_t input_size, size_t output_size = 0) const {
        if (output_size == 0) {
            output_size = input_size;
        }
        double total_bytes = num_elements * (input_size + output_size); // H2D + D2H
        double total_gb = total_bytes / 1e9;
        
        double total_samples_mega = num_elements / 1e6; 

        double kernel_sec = compute_ms / 1000.0;
        double total_sec = total_ms / 1000.0;

        double cold_sec   = (initialization_ms + total_ms) / 1000.0;

        std::cout << std::fixed << std::setprecision(3);
        
        std::cout << "1. LATENCY BREAKDOWN (Steady State)" << std::endl;
        if (transfer_h2d_ms > 0) std::cout << "   H2D Transfer:   " << transfer_h2d_ms << " ms" << std::endl;
        std::cout << "   Kernel Compute: " << compute_ms << " ms" << std::endl;
        if (transfer_d2h_ms > 0) std::cout << "   D2H Transfer:   " << transfer_d2h_ms << " ms" << std::endl;
        std::cout << "   -----------------------------" << std::endl;
        std::cout << "   TOTAL LATENCY:  " << total_ms << " ms" << std::endl;

        std::cout << "\n2. THROUGHPUT (Steady State)" << std::endl;
        if (compute_ms > 0) {
            std::cout << "   Kernel Bandwidth: " << (total_gb / kernel_sec) << " GB/s" << std::endl;
            std::cout << "   Kernel Speed:   " << (total_samples_mega / kernel_sec) << " Mega Samples/s" << std::endl;
        }
        std::cout << "   App BandWidth:   " << (total_gb / total_sec) << " GB/s" << std::endl;
        std::cout << "   App Speed:      " << (total_samples_mega / total_sec) << " Mega Samples/s" << std::endl;
        std::cout << "   Cold Start:     " << (total_samples_mega / cold_sec) << " Mega Samples/s (Includes Init)" << std::endl;

        std::cout << "\n3. INITIALIZATION COST (One-time)" << std::endl;
        std::cout << "   Allocation:     " << initialization_ms << " ms" << std::endl;
        std::cout << "   First Frame:    " << (initialization_ms + total_ms) << " ms (Cold Start)" << std::endl;
        std::cout << "___________________________________\n" << std::endl;
    }
};

class GpuTimer {
    cudaEvent_t start_evt, h2d_evt, compute_evt, stop_evt;
public:
    GpuTimer() {
        cudaEventCreate(&start_evt); cudaEventCreate(&h2d_evt);
        cudaEventCreate(&compute_evt); cudaEventCreate(&stop_evt);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_evt); cudaEventDestroy(h2d_evt);
        cudaEventDestroy(compute_evt); cudaEventDestroy(stop_evt);
    }
    void start() { cudaEventRecord(start_evt); } 
    void mark_h2d() { cudaEventRecord(h2d_evt); } 
    void mark_compute() { cudaEventRecord(compute_evt); } 
    void stop() { cudaEventRecord(stop_evt); cudaEventSynchronize(stop_evt); } 
 
    ProfileResult get_result() {
        ProfileResult res;
        cudaEventElapsedTime(&res.transfer_h2d_ms, start_evt, h2d_evt);
        cudaEventElapsedTime(&res.compute_ms, h2d_evt, compute_evt);
        cudaEventElapsedTime(&res.transfer_d2h_ms, compute_evt, stop_evt);
        res.total_ms = res.transfer_h2d_ms + res.compute_ms + res.transfer_d2h_ms;
        return res;
    }
};

class CpuTimer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> t_start, t_end;
public:
    void start() { t_start = Clock::now(); }
    void mark_h2d() {} 
    void mark_compute() {} 
    void stop() { t_end = Clock::now(); }

    ProfileResult get_result() {
        ProfileResult res;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
        res.compute_ms = duration.count() / 1000.0f;
        res.total_ms = res.compute_ms;
        return res;
    }
};

template <typename TimerType, typename Func>
ProfileResult benchmark(int iterations, int warmup, Func pipeline_lambda) {
    TimerType timer;

    for(int i=0; i<warmup; ++i) {
        pipeline_lambda(timer); 
    }

    ProfileResult avg;
    for(int i=0; i<iterations; ++i) {
        pipeline_lambda(timer);
        avg += timer.get_result();
    }
    
    avg.divide(iterations);
    return avg;
}