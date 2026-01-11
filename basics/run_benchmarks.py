import subprocess
import os
import numpy as np
import scipy.io.wavfile as wav
import time
import gc # Garbage Collector

EXECUTABLES = [
    {"name": "CPU_SingleThread", "path": "./bin_cpu",      "needs_block": False},
    {"name": "Parallel_Avg", "path": "./bin_parallel",      "needs_block": True},
    {"name": "SharedMem",    "path": "./bin_shared",   "needs_block": True},
    {"name": "Vectorized_int2",   "path": "./bin_vec2",      "needs_block": True},
    {"name": "Vectorized_int4",   "path": "./bin_vec4",      "needs_block": True},
    {"name": "HillisSteele", "path": "./bin_hillis",   "needs_block": True},
    {"name": "V_HillisSteele", "path": "./bin_vhillis",   "needs_block": True},
    {"name": "Blelloch",     "path": "./bin_blelloch", "needs_block": True},
    {"name": "V_Blelloch",     "path": "./bin_vblelloch", "needs_block": True},
]


BLOCK_SIZES = [32, 64, 128, 256, 512, 1024] 

GRADES = list(range(1, 11, 1)) + list(range(11, 51, 5)) + list(range(50, 1001, 50))

# 100M samples is around the upper limit for Hillis-Steele on 4GB RAM.
INPUT_SIZES = np.linspace(5000, 50000000, 100, dtype=int)

TEMP_WAV = "temp_bench.wav"


def generate_wav(num_samples, channels=2):
    """
    Generates a random Stereo pcm 16 bit WAV file.
    """
    print(f"Allocating RAM for {num_samples} samples")
    num_samples = int(num_samples / 2);  
    try:
        data = np.random.randint(-32768, 32767, size=(num_samples, channels), dtype=np.int16)
        
        print("Writing to disk")
        wav.write(TEMP_WAV, 44100, data)
        
        del data
        gc.collect()
        print("Generation Complete")
        return True
    except MemoryError:
        print(f"Could not generate {num_samples} samples.")
        return False

def run_suite():
    total_runs_estimate = len(INPUT_SIZES) * len(GRADES) * len(BLOCK_SIZES) * len(EXECUTABLES)
    print(f"__________________________")
    print(f"STARTING BENCHMARK SUITE")
    print(f"Target Runs: about {total_runs_estimate}")
    print(f"Max Samples: {INPUT_SIZES[-1]} (100M)")
    print(f"__________________________\n")
    
    start_time = time.time()
    counter = 0
    failures = 0

    for n_samples in INPUT_SIZES:
        print(f"\nInput Size: {n_samples}")
        
        if not generate_wav(n_samples):  
            print("Skipping this input size due to generation failure.")
            continue
        
        for exe in EXECUTABLES:
            
            if not os.path.exists(exe["path"]):
                print(f"Binary not found: {exe['path']}")
                continue

            for grade in GRADES:
                
                if grade >= n_samples: 
                    continue

                current_blocks = BLOCK_SIZES if exe["needs_block"] else [256] 
                
                for b_size in current_blocks:
                    counter += 1
                    
                    cmd = [exe["path"], TEMP_WAV, str(grade), str(b_size)]
                    
                    try:
                        # Here I'm using subprocess to isolate the C++ memory space from Python
                        
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                        
                        if result.returncode != 0:
                            failures += 1
                            print(f"Failure: {exe['name']} (N={n_samples}, G={grade}, B={b_size})")
                            print(f"Return Code: {result.returncode}")
                            print(f"Error: {result.stderr}")
                        else:
                            pass

                    except Exception as e:
                        print(f"Python execution failed: {e}")

                    if counter % 50 == 0:
                        elapsed = time.time() - start_time
                        print(f"{counter} runs. Elapsed: {elapsed:.1f}s")

    if os.path.exists(TEMP_WAV):
        os.remove(TEMP_WAV)
        
    print("\n__________________________________"
    print(f"BENCHMARK COMPLETE")
    print(f"Total Runs: {counter}")
    print(f"Total Failures/Crashes: {failures}")
    print(f"Results saved to: benchmark_results.csv")
    print(f"__________________________________\n")

if __name__ == "__main__":
    run_suite()


# Compile the C++ files:

# nvcc -O3 profilable_moving_averager.cpp -o bin_cpu

# nvcc -O3 profilable_parallel_averager.cu -o bin_parallel

# nvcc -O3 profilable_sm_averager.cu -o bin_shared

# nvcc -O3 profilable_sm_vload2.cu -o bin_vec2

# nvcc -O3 profilable_sm_vload4.cu -o bin_vec4

# nvcc -O3 hillis_steele_averager.cu -o bin_hillis

# nvcc -O3 hillis_steele_vloaded_averager.cu -o bin_vhillis

# nvcc -O3 blelloch_scan_averager.cu -o bin_blelloch

# nvcc -O3 blelloch_scan_vloaded_averager.cu -o bin_vblelloch