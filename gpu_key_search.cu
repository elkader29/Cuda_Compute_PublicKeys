#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <curand_kernel.h>
#include <chrono>
#include "secp256k1/inc_vendor.h"
#include "secp256k1/inc_types.h"
#include "secp256k1/inc_platform.hh"
#include "secp256k1/inc_common.h"
#include "secp256k1/inc_ecc_secp256k1.h"
#include "sha256.h"
#include "ripemd160.h"

// ===================== Utility: 256-bit integer (little-endian limbs) =====================

struct U256 {
    // Little-endian limbs: v[0] is least-significant 64 bits
    uint64_t v[4];

    static U256 from_hex(const std::string& hex) {
        std::string s = hex;
        if (s.rfind("0x",0)==0 || s.rfind("0X",0)==0) s = s.substr(2);

        // Pad to 64 characters (256 bits)
        if (s.length() < 64) {
            s = std::string(64 - s.length(), '0') + s;
        }

        U256 out;
        for (int i = 0; i < 4; ++i) {
            std::string limb_str = s.substr((3 - i) * 16, 16);
            out.v[i] = std::stoull(limb_str, nullptr, 16);
        }
        return out;
    }

    std::string to_hex() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (int i = 3; i >= 0; --i) {
            oss << std::setw(16) << v[i];
        }
        return oss.str();
    }

    // Compare (return -1,0,1)
    __host__ __device__ int cmp(const U256& b) const {
        for (int i=3;i>=0;--i){
            if (v[i]<b.v[i]) return -1;
            if (v[i]>b.v[i]) return 1;
        }
        return 0;
    }

    // this = this + b
    __host__ __device__ void add(const U256& b) {
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 sum = (unsigned __int128)v[i] + b.v[i] + carry;
            v[i] = (uint64_t)sum;
            carry = sum >> 64;
        }
    }

    // returns a = a - b
    __host__ __device__ void sub(const U256& b) {
        unsigned __int128 borrow = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 diff = (unsigned __int128)v[i] - b.v[i] - borrow;
            v[i] = (uint64_t)diff;
            borrow = (diff >> 64) & 1;
        }
    }

    // Fill with random data from CURAND
    __device__ void generate_random(curandState* state) {
        v[0] = ((uint64_t)curand(state) << 32) | curand(state);
        v[1] = ((uint64_t)curand(state) << 32) | curand(state);
        v[2] = ((uint64_t)curand(state) << 32) | curand(state);
        v[3] = ((uint64_t)curand(state) << 32) | curand(state);
    }

    // Serialize to big-endian 32-byte array (for private key)
    __device__ void to_priv_key_bytes(uint32_t* out) {
        // U256 limbs are little-endian, private key needs to be big-endian
        out[0] = hc_swap32_S((uint32_t)(v[3] >> 32));
        out[1] = hc_swap32_S((uint32_t)(v[3]));
        out[2] = hc_swap32_S((uint32_t)(v[2] >> 32));
        out[3] = hc_swap32_S((uint32_t)(v[2]));
        out[4] = hc_swap32_S((uint32_t)(v[1] >> 32));
        out[5] = hc_swap32_S((uint32_t)(v[1]));
        out[6] = hc_swap32_S((uint32_t)(v[0] >> 32));
        out[7] = hc_swap32_S((uint32_t)(v[0]));
    }
};

// ===================== CLI parsing =====================

struct Args {
    U256 start, end;
    std::string target;
    std::string outputFile;
    bool randomMode = false;
    int seconds = 0;
};

static void usage(const char* prog){
    std::cerr <<
      "Usage: " << prog << " --keyspace <hex_start:hex_end> -t <target_or_file> --random [-o <output.bin>] [--seconds <t>]\n"
      "  --keyspace <start:end> : Required. Hexadecimal range for private keys.\n"
      "  -t <target>            : Required. Target public key (hex) or file with keys.\n"
      "  -R, --random           : Required. Use random generation mode.\n"
      "  -o <file>              : Required. File to write private keys and fingerprints to.\n";
}

static bool parse_args(int argc, char** argv, Args& a){
    bool keyspace_set = false;
    bool target_set = false;
    bool output_set = false;

    for (int i=1;i<argc;++i){
        std::string s = argv[i];
        if ((s == "--keyspace") && i+1<argc){
            std::string r = argv[++i];
            auto pos = r.find(':');
            if (pos==std::string::npos){ std::cerr<<"Bad --keyspace format. Expected <hex_start:hex_end>\n"; return false; }
            a.start = U256::from_hex(r.substr(0,pos));
            a.end   = U256::from_hex(r.substr(pos+1));
            keyspace_set = true;
        } else if ((s == "-t") && i+1<argc){
            a.target = argv[++i];
            target_set = true;
        } else if (s == "-o" && i+1<argc) {
            a.outputFile = argv[++i];
            output_set = true;
        } else if (s == "--random" || s == "-R") {
            a.randomMode = true;
        } else if (s == "--seconds" && i+1<argc) {
            a.seconds = std::atoi(argv[++i]);
        } else {
            std::cerr<<"Unknown or malformed arg: "<<s<<"\n"; usage(argv[0]); return false;
        }
    }

    if (!keyspace_set || !target_set || !a.randomMode || !output_set) {
        std::cerr << "Error: --keyspace, -t, -o, and --random/-R are all required.\n";
        usage(argv[0]);
        return false;
    }
    if (a.start.cmp(a.end) > 0){ std::cerr<<"Error: start of keyspace is greater than end.\n"; return false; }
    return true;
}

// ===================== Target Loading =====================

using fingerprint_t = std::array<uint8_t, 24>;

// Helper to convert hex string to bytes
static std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    std::string hex_copy = hex;
    if (hex_copy.rfind("0x",0)==0 || hex_copy.rfind("0X",0)==0) hex_copy = hex_copy.substr(2);

    if (hex_copy.length() % 2 != 0) {
        hex_copy = "0" + hex_copy;
    }
    for (unsigned int i = 0; i < hex_copy.length(); i += 2) {
        std::string byteString = hex_copy.substr(i, 2);
        uint8_t byte = (uint8_t) strtol(byteString.c_str(), nullptr, 16);
        bytes.push_back(byte);
    }
    return bytes;
}


static bool load_targets(const std::string& target_arg, std::vector<fingerprint_t>& targets) {
    auto process_key = [&](std::string key_hex) {
        // Trim whitespace
        auto first = key_hex.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) return;
        auto last = key_hex.find_last_not_of(" \t\n\r");
        key_hex = key_hex.substr(first, (last - first + 1));

        if (key_hex.length() != 66) {
            //std::cerr << "Warning: Invalid public key length (" << key_hex.length() << "), skipping: " << key_hex << std::endl;
            return;
        }
        std::vector<uint8_t> pubkey_bytes = hex_to_bytes(key_hex);
        if (pubkey_bytes.size() != 33) {
             //std::cerr << "Warning: Invalid public key bytes after hex conversion, skipping: " << key_hex << std::endl;
            return;
        }

        fingerprint_t fingerprint;
        // Compressed pubkey is [1-byte prefix][32-byte X]. Last 24 bytes of X are at offset 9.
        std::copy_n(pubkey_bytes.begin() + 9, 24, fingerprint.begin());
        targets.push_back(fingerprint);
    };

    std::ifstream target_file(target_arg);
    if (target_file.is_open()) {
        std::cout << "Reading targets from file: " << target_arg << std::endl;
        std::string line;
        while (std::getline(target_file, line)) {
            if (!line.empty()) {
                process_key(line);
            }
        }
    } else {
        std::cout << "Reading single target from command line." << std::endl;
        process_key(target_arg);
    }

    if (targets.empty()) {
        std::cerr << "Error: No valid targets were loaded from '" << target_arg << "'\n";
        return false;
    }

    std::cout << "Loaded " << targets.size() << " target fingerprints." << std::endl;
    return true;
}


__constant__ secp256k1_t s_basepoint;
unsigned int BLOCK_THREADS = 512; // Maximize the threads per block
unsigned int BLOCK_NUMBER = 0; // Set based on GPU properties dynamically

#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        cudaDeviceReset(); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void init_curand_kernel(curandState* state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

#ifndef DEV_FAKE_ECC
#define DEV_FAKE_ECC 1
#endif

// Device function to compute the compressed public key
__device__ void private_to_public(const uint32_t* pri, uint32_t* pub) {
#if DEV_FAKE_ECC
    // Use SHA256 as a stand-in for ECC multiplication to test the pipeline
    unsigned char hash[SHA256_BLOCK_SIZE];
    SHA256::hash((const unsigned char*)pri, 32, hash);
    // Copy hash to pub to simulate a 33-byte public key (we'll just use 32 bytes of it)
    for(int i=0; i<8; ++i) {
        pub[i+1] = ((uint32_t*)hash)[i];
    }
    pub[0] = 0x02000000; // Fake a prefix
#else
    uint32_t a[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        a[i] = hc_swap32_S(pri[7 - i]);

    point_mul(pub, a, &s_basepoint);

    #pragma unroll
    for (int i = 0; i < 9; i++)
        pub[i] = hc_swap32_S(pub[i]);
#endif
}

// Device function to extract the HASH160 fingerprint from a compressed public key
__device__ void get_fingerprint(const uint32_t* pubKey, uint32_t* fingerprint) {
    // HASH160 is RIPEMD160(SHA256(data))
    unsigned char sha_hash[SHA256_BLOCK_SIZE];
    SHA256::hash((const unsigned char*)pubKey, 33, sha_hash);

    unsigned char ripemd_hash[RIPEMD160_BLOCK_SIZE];
    RIPEMD160::hash(sha_hash, SHA256_BLOCK_SIZE, ripemd_hash);

    // Copy 20-byte hash to 24-byte fingerprint buffer, zero-padding the rest
    for(int i=0; i<5; ++i) { // 5 * 4 bytes = 20 bytes
        fingerprint[i] = ((uint32_t*)ripemd_hash)[i];
    }
    fingerprint[5] = 0; // Zero out the last 4 bytes
}

// Kernel to generate a private key and compute compressed public key
__global__ void generate_keypair_kernel(curandState* state, uint32_t* prvKeys, uint32_t* compressedPubKeys, uint32_t* fingerprints, uint32_t* found_priv_keys, unsigned int* found_count, U256 key_start, U256 range_size, const uint32_t* target_fingerprints, unsigned int target_count) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[id];

    // --- Ranged Random Key Generation ---
    __shared__ U256 curve_order_n;
    if (threadIdx.x == 0) {
        // The order N of the secp256k1 curve
        curve_order_n.v[3] = 0xFFFFFFFFFFFFFFFF;
        curve_order_n.v[2] = 0xFFFFFFFFFFFFFFFE;
        curve_order_n.v[1] = 0xBAAEDCE6AF48A03B;
        curve_order_n.v[0] = 0xBFD25E8CD0364141;
    }
    __syncthreads();

    U256 private_key;
    U256 zero = {0,0,0,0}; // a zeroed key for comparison
    while(true) {
        U256 random_offset;
        random_offset.generate_random(&localState);
        if(random_offset.cmp(range_size) < 0) {
            private_key = key_start;
            private_key.add(random_offset);
            // Private key must be > 0 and < n to be valid
            if (private_key.cmp(zero) > 0 && private_key.cmp(curve_order_n) < 0) {
                 break;
            }
        }
    }
    // --- End Ranged Random Key Generation ---

    uint32_t* p = &prvKeys[id * 8];
    private_key.to_priv_key_bytes(p);

    // Compute the compressed public key
    uint32_t* pub = &compressedPubKeys[id * 9];
    private_to_public(p, pub);

    // Extract the fingerprint
    uint32_t* fp = &fingerprints[id * 6];
    get_fingerprint(pub, fp);

    // Compare against targets
    for(unsigned int i = 0; i < target_count; ++i) {
        bool match = true;
        const uint32_t* target_fp = &target_fingerprints[i * 6];
        #pragma unroll
        for(int j = 0; j < 6; ++j) {
            if(fp[j] != target_fp[j]) {
                match = false;
                break;
            }
        }
        if(match) {
            unsigned int result_idx = atomicAdd(found_count, 1);
            if(result_idx < 100) { // Only store if we have space in the buffer
                // U256 private_key needs to be written to found_priv_keys
                // private_key is already in the correct format in `p`
                uint32_t* dest = &found_priv_keys[result_idx * 8];
                #pragma unroll
                for(int k=0; k<8; ++k) {
                    dest[k] = p[k];
                }
            }
        }
    }

    state[id] = localState;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    if (BLOCK_NUMBER == 0) {
        BLOCK_NUMBER = props.multiProcessorCount * 4;  // Set a high number of blocks to keep the GPU busy
    }

    fprintf(stderr, "[!] %s (%2d procs | Blocks: %d | Threads: %d)\n", props.name, props.multiProcessorCount, BLOCK_NUMBER, BLOCK_THREADS);

    secp256k1_t basepoint;
    set_precomputed_basepoint_g(&basepoint);
    cudaMemcpyToSymbol(s_basepoint, &basepoint, sizeof(basepoint));

    // Calculate range size and copy to device
    U256 range_size = args.end;
    range_size.sub(args.start);
    U256 one;
    one.v[0] = 1;
    range_size.add(one);

    // Load targets from host
    std::vector<fingerprint_t> host_targets;
    if(!load_targets(args.target, host_targets)) {
        return 1;
    }

    // Allocate and copy targets to device global memory
    uint32_t* d_target_fingerprints;
    unsigned int target_count = host_targets.size();
    size_t targets_size_bytes = target_count * sizeof(fingerprint_t);
    cudaMalloc(&d_target_fingerprints, targets_size_bytes);
    cudaMemcpy(d_target_fingerprints, host_targets.data(), targets_size_bytes, cudaMemcpyHostToDevice);

    // Result buffer for matches
    uint32_t* d_found_priv_keys;
    unsigned int* d_found_count;
    cudaMalloc(&d_found_priv_keys, 100 * 8 * sizeof(uint32_t)); // Buffer for up to 100 matches
    cudaMalloc(&d_found_count, sizeof(unsigned int));
    cudaMemset(d_found_count, 0, sizeof(unsigned int));

    // Open matches file
    std::ofstream matches_file("matches.bin", std::ios::binary | std::ios::app);
    if (!matches_file) {
        std::cerr << "Error: Could not open matches.bin for writing." << std::endl;
        return 1;
    }

    curandState* d_state;
    uint32_t* d_prvKeys;
    uint32_t* d_compressedPubKeys;
    uint32_t* d_fingerprints;
    size_t totalThreads = BLOCK_NUMBER * BLOCK_THREADS;
    cudaMalloc((void**)&d_prvKeys, 8 * totalThreads * sizeof(uint32_t));
    cudaMalloc((void**)&d_compressedPubKeys, 9 * totalThreads * sizeof(uint32_t));
    cudaMalloc((void**)&d_fingerprints, 6 * totalThreads * sizeof(uint32_t)); // 24 bytes per fingerprint
    cudaMalloc((void**)&d_state, totalThreads * sizeof(curandState));
    cudaCheckError();

    init_curand_kernel<<<BLOCK_NUMBER, BLOCK_THREADS>>>(d_state, time(0));
    cudaCheckError();

    // Variables for performance measurement
    auto start = std::chrono::high_resolution_clock::now();
    unsigned long long totalKeysGenerated = 0;

    // Open output file
    std::ofstream output_db_file(args.outputFile, std::ios::binary | std::ios::app);
    if (!output_db_file) {
        std::cerr << "Error: Could not open " << args.outputFile << " for writing." << std::endl;
        return 1;
    }

    // Host buffers
    std::vector<uint32_t> h_prvKeys(totalThreads * 8);
    std::vector<uint32_t> h_fingerprints(totalThreads * 6);

    // Loop to continuously generate keys and write to db
    auto overall_start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        // Launch the kernel to generate keys
        generate_keypair_kernel<<<BLOCK_NUMBER, BLOCK_THREADS>>>(d_state, d_prvKeys, d_compressedPubKeys, d_fingerprints, d_found_priv_keys, d_found_count, args.start, range_size, d_target_fingerprints, target_count);
        cudaCheckError();

        // Copy results back to host
        cudaMemcpy(h_prvKeys.data(), d_prvKeys, h_prvKeys.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fingerprints.data(), d_fingerprints, h_fingerprints.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Write to database file
        for(size_t i = 0; i < totalThreads; ++i) {
            output_db_file.write(reinterpret_cast<const char*>(&h_prvKeys[i * 8]), 32); // 32-byte private key
            output_db_file.write(reinterpret_cast<const char*>(&h_fingerprints[i * 6]), 24); // 24-byte fingerprint
        }

        // Check for matches
        unsigned int h_found_count = 0;
        cudaMemcpy(&h_found_count, d_found_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (h_found_count > 0) {
            std::cout << "\nFound " << h_found_count << " matches!" << std::endl;
            std::vector<uint32_t> h_found_keys(h_found_count * 8);
            cudaMemcpy(h_found_keys.data(), d_found_priv_keys, h_found_keys.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            for(unsigned int i = 0; i < h_found_count; ++i) {
                matches_file.write(reinterpret_cast<const char*>(&h_found_keys[i*8]), 32);
            }
            matches_file.flush();

            // Reset counter
            cudaMemset(d_found_count, 0, sizeof(unsigned int));
        }

        totalKeysGenerated += totalThreads;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if(elapsed.count() > 2.0) {
            double keysPerSecond = totalKeysGenerated / elapsed.count();
            printf("Speed: %.2f Mkeys/s\n", keysPerSecond / 1e6);
            fflush(stdout);
            totalKeysGenerated = 0;
            start = end;
        }

        // Check for program duration
        if(args.seconds > 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            double total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - overall_start_time).count();
            if(total_elapsed >= args.seconds) {
                printf("Finished after %.2f seconds.\n", total_elapsed);
                break;
            }
        }
    }

    // Cleanup (not reached due to infinite loop)
    cudaFree(d_prvKeys);
    cudaFree(d_compressedPubKeys);
    cudaFree(d_fingerprints);
    cudaFree(d_state);
    cudaFree(d_found_priv_keys);
    cudaFree(d_found_count);
    cudaFree(d_target_fingerprints);

    return 0;
}
