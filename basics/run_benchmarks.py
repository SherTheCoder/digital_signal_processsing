import subprocess
import os
import numpy as np
import scipy.io.wavfile as wav
import time
import gc # Garbage Collector interface

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# List of your compiled executables
# 'needs_block': True for GPU algorithms, False for CPU (skips block size loop)
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

# 1. Block Sizes to test (Standard GPU sizes)
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024] 

# 2. Grades (Window Sizes) to test
GRADES = list(range(1, 11, 1)) + list(range(11, 51, 5)) + list(range(50, 1001, 50))

# 3. Input Sizes (Number of Samples)
# Range: 5k to 50 Million
# Note: 100M samples is the "Danger Zone" for Hillis-Steele on 4GB RAM.
INPUT_SIZES = np.linspace(5000, 50000000, 100, dtype=int)

# Temporary file to pass data between Python and C++
TEMP_WAV = "temp_bench.wav"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def generate_wav(num_samples, channels=2):
    """
    Generates a random Stereo PCM 16-bit WAV file.
    Uses 'try-except' to catch Python-side OOM before crashing the OS.
    """
    print(f"   -> Allocating RAM for {num_samples} samples...")
    num_samples = int(num_samples / 2);  # Stereo adjustment
    try:
        # Generate random int16 noise
        # Shape (N, 2) ensures Stereo header is written correctly
        data = np.random.randint(-32768, 32767, size=(num_samples, channels), dtype=np.int16)
        
        print("   -> Writing to disk...")
        wav.write(TEMP_WAV, 44100, data)
        
        # Explicitly delete data and run GC to free Python RAM immediately
        del data
        gc.collect()
        print("   -> Generation Complete.")
        return True
    except MemoryError:
        print(f"   [CRITICAL] Python OOM: Could not generate {num_samples} samples.")
        return False

def run_suite():
    total_runs_estimate = len(INPUT_SIZES) * len(GRADES) * len(BLOCK_SIZES) * len(EXECUTABLES)
    print(f"================================================================")
    print(f"STARTING BENCHMARK SUITE")
    print(f"Target Runs: ~{total_runs_estimate}")
    print(f"Max Samples: {INPUT_SIZES[-1]} (100M)")
    print(f"================================================================\n")
    
    start_time = time.time()
    counter = 0
    failures = 0

    # --- OUTER LOOP: INPUT SIZE ---
    for n_samples in INPUT_SIZES:
        print(f"\n[SET] Input Size: {n_samples}")
        
        # 1. Generate Test File
        if not generate_wav(n_samples):  # Divide by 2 for Stereo
            print("Skipping this input size due to generation failure.")
            continue
        
        # --- MIDDLE LOOP: ALGORITHMS ---
        for exe in EXECUTABLES:
            
            # Check if binary exists
            if not os.path.exists(exe["path"]):
                print(f"   [SKIP] Binary not found: {exe['path']}")
                continue

            # --- INNER LOOP: GRADES ---
            for grade in GRADES:
                
                # Logic Check: Window size cannot be larger than the file itself
                if grade >= n_samples: 
                    continue

                # --- INNER LOOP: BLOCK SIZES ---
                # If CPU, we run only once (BlockSize 0 or 1, ignored by C++)
                current_blocks = BLOCK_SIZES if exe["needs_block"] else [256] 
                
                for b_size in current_blocks:
                    counter += 1
                    
                    # Prepare Command: ./exe <wav> <grade> <block_size>
                    cmd = [exe["path"], TEMP_WAV, str(grade), str(b_size)]
                    
                    try:
                        # Run the C++ executable
                        # We use subprocess to isolate the C++ memory space from Python
                        
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                        
                        if result.returncode != 0:
                            # Non-zero return means Crash, Segfault, or OOM Kill
                            failures += 1
                            print(f"   [FAIL] {exe['name']} (N={n_samples}, G={grade}, B={b_size})")
                            print(f"          Return Code: {result.returncode}")
                            # Optional: Print stderr for debugging
                            print(f"          Error: {result.stderr}")
                        else:
                            # Success is silent to keep console clean (C++ logs to CSV)
                            pass

                    except Exception as e:
                        print(f"   [ERROR] Python execution failed: {e}")

                    # Status Update every 50 runs
                    if counter % 50 == 0:
                        elapsed = time.time() - start_time
                        print(f"   ... Progress: {counter} runs. Elapsed: {elapsed:.1f}s")

    # --- CLEANUP ---
    if os.path.exists(TEMP_WAV):
        os.remove(TEMP_WAV)
        
    print("\n================================================================")
    print(f"BENCHMARK COMPLETE")
    print(f"Total Runs: {counter}")
    print(f"Total Failures/Crashes: {failures}")
    print(f"Results saved to: benchmark_results.csv")
    print(f"================================================================")

if __name__ == "__main__":
    run_suite()



# # 1. CPU Single Thread
# nvcc -O3 profilable_moving_averager.cpp -o bin_cpu

# # 2. Basic Parallel
# nvcc -O3 profilable_parallel_averager.cu -o bin_parallel

# # 3. Shared Memory
# nvcc -O3 profilable_sm_averager.cu -o bin_shared

# # 4. Vectorized int2
# nvcc -O3 profilable_sm_vload2.cu -o bin_vec2

# # 5. Vectorized int4
# nvcc -O3 profilable_sm_vload4.cu -o bin_vec4

# # 6. Hillis Steele (Scalar)
# nvcc -O3 hillis_steele_averager.cu -o bin_hillis

# # 7. Hillis Steele (Vectorized)
# nvcc -O3 hillis_steele_vloaded_averager.cu -o bin_vhillis

# # 8. Blelloch (Scalar)
# nvcc -O3 blelloch_scan_averager.cu -o bin_blelloch

# # 9. Blelloch (Vectorized)
# nvcc -O3 blelloch_scan_vloaded_averager.cu -o bin_vblelloch