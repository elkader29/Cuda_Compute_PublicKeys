// gpu_search.cu
// A high-performance tool using CUDA to search for secp256k1 public keys.
// It generates private keys in a given range, derives compressed public keys,
// and compares their 24-byte fingerprints against a list of targets.
//
// Build: nvcc -O3 -std=c++17 gpu_search.cu -lssl -lcrypto -o gpu_search
// Tested with CUDA 11+/OpenSSL 1.1+/3.x

#include <cuda.h>
#include <curand_kernel.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>
#include <openssl/bn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <fstream>
#include <map>
#include <algorithm>
#include <chrono>

// ===================== Utility: 256-bit integer (little-endian limbs) =====================

struct U256 {
    // Little-endian limbs: v[0] is least-significant 64 bits
    uint64_t v[4];

    U256() { v[0]=v[1]=v[2]=v[3]=0; }

    static U256 from_hex(const std::string& hex) {
        std::string s = hex;
        if (s.rfind("0x",0)==0 || s.rfind("0X",0)==0) s = s.substr(2);
        // pad to even length
        if (s.size()%2) s = "0"+s;
        // parse into bytes big-endian, then pack little-endian limbs
        std::vector<uint8_t> bytes(s.size()/2);
        for (size_t i=0;i<bytes.size();++i){
            unsigned x;
            std::sscanf(s.substr(2*i,2).c_str(), "%02x", &x);
            bytes[i] = static_cast<uint8_t>(x);
        }
        // pack
        U256 out;
        size_t bi = 0;
        // write bytes from end (big-endian) into little-endian limbs
        for (int limb=0; limb<4; ++limb) {
            uint64_t w=0;
            for (int j=0;j<8;++j){
                size_t srcIndex = bytes.size() > bi ? bytes.size()-1 - bi : SIZE_MAX;
                uint8_t b = (srcIndex<bytes.size()) ? bytes[srcIndex] : 0;
                w |= (uint64_t)b << (j*8);
                ++bi;
            }
            out.v[limb] = w;
        }
        return out;
    }

    std::string to_hex() const {
        std::ostringstream oss;
        // print big-endian
        for (int limb=3; limb>=0; --limb) {
            oss << std::hex << std::setw(16) << std::setfill('0') << std::nouppercase << v[limb];
        }
        std::string s = oss.str();
        // strip leading zeros but keep at least one '0'
        size_t p = s.find_first_not_of('0');
        if (p==std::string::npos) return "0";
        return "0x"+s.substr(p);
    }

    // Compare (return -1,0,1)
    int cmp(const U256& b) const {
        for (int i=3;i>=0;--i){
            if (v[i]<b.v[i]) return -1;
            if (v[i]>b.v[i]) return 1;
        }
        return 0;
    }

    // this += x (x fits in 64 bits), return carry
    __host__ __device__ uint64_t add_u64(uint64_t x){
        unsigned __int128 t = (unsigned __int128)v[0] + x;
        v[0] = (uint64_t)t;
        uint64_t c = (uint64_t)(t>>64);
        for (int i=1;i<4 && c;i++){
            unsigned __int128 t2 = (unsigned __int128)v[i] + c;
            v[i] = (uint64_t)t2;
            c = (uint64_t)(t2>>64);
        }
        return c;
    }

    // out = a + idx (idx 64-bit), return carry
    __host__ __device__ static uint64_t add_u64(const U256& a, uint64_t idx, U256& out){
        unsigned __int128 t = (unsigned __int128)a.v[0] + idx;
        out.v[0] = (uint64_t)t;
        uint64_t c = (uint64_t)(t>>64);
        for (int i=1;i<4;i++){
            unsigned __int128 t2 = (unsigned __int128)a.v[i] + c;
            out.v[i] = (uint64_t)t2;
            c = (uint64_t)(t2>>64);
        }
        return c;
    }

    // out = a + b (b fits into 64-bit), but here we only need add_u64.
};

// pack U256 to 32 bytes big-endian
static inline void u256_to_bytes_be(const U256& x, uint8_t out[32]){
    for (int limb=3; limb>=0; --limb){
        uint64_t w = x.v[limb];
        for (int j=7;j>=0;--j){
            out[(3-limb)*8 + (7-j)] = (uint8_t)((w >> (j*8)) & 0xFF);
        }
    }
}

// add small (<= 64-bit) to U256: host helper
static inline U256 add_u256_u64(const U256& a, uint64_t add){
    U256 o=a;
    o.add_u64(add);
    return o;
}

// pick a random base inside [start, end-count+1]
static U256 random_base_in_range(const U256& start, const U256& end, uint64_t count) {
    // We'll do: base = start + r, where r is uniform in [0, range-count]
    // Implement simple rejection over 256-bit using std::random_device
    U256 range = end; // range = end - start + 1
    // compute end - start + 1
    // using 256-bit subtraction:
    U256 s=start, e=end;
    // e = end - start
    uint64_t borrow=0;
    for (int i=0;i<4;++i){
        unsigned __int128 a = (unsigned __int128)e.v[i];
        unsigned __int128 b = (unsigned __int128)s.v[i] + borrow;
        if (a >= b) { e.v[i] = (uint64_t)(a-b); borrow=0; }
        else { e.v[i] = (uint64_t)((((unsigned __int128)1<<64) + a) - b); borrow=1; }
    }
    // e = e + 1
    e.add_u64(1);

    // subtract (count) from e to get max r+1
    // e = e - count
    borrow=0;
    unsigned __int128 c = (unsigned __int128)(count);
    unsigned __int128 a0 = (unsigned __int128)e.v[0];
    if (a0 >= c) { e.v[0] = (uint64_t)(a0 - c); borrow=0; }
    else { e.v[0] = (uint64_t)((((unsigned __int128)1<<64) + a0) - c); borrow=1; }
    for (int i=1;i<4;++i){
        unsigned __int128 ai = (unsigned __int128)e.v[i];
        if (ai >= borrow) { e.v[i] = (uint64_t)(ai - borrow); borrow=0; }
        else { e.v[i] = (uint64_t)((((unsigned __int128)1<<64) + ai) - borrow); borrow=1; }
    }
    // Now e is (range - count + 1): the number of valid starting positions.

    std::random_device rd;
    std::mt19937_64 gen(rd());
    auto rnd64 = [&](){ return ((unsigned __int128)gen() << 64) ^ gen(); };

    // sample r in [0, e-1]
    U256 r;
    while (true) {
        // fill r with 128+128 bits
        unsigned __int128 R0 = rnd64();
        unsigned __int128 R1 = rnd64();
        r.v[0] = (uint64_t)R0;
        r.v[1] = (uint64_t)(R0>>64);
        r.v[2] = (uint64_t)R1;
        r.v[3] = (uint64_t)(R1>>64);

        // if r < e, accept
        if (r.cmp(e) < 0) break;
    }

    // base = start + r
    U256 base=start;
    // base += r (full 256-bit add)
    unsigned __int128 carry=0;
    for (int i=0;i<4;++i){
        unsigned __int128 t = (unsigned __int128)base.v[i] + r.v[i] + carry;
        base.v[i] = (uint64_t)t;
        carry = t>>64;
    }
    return base;
}

// ===================== CUDA: fill sequential or random within a window =====================

__global__ void fill_seq_kernel(const U256 base, uint8_t* priv32_out, uint64_t count) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // priv = base + idx
    U256 p;
    U256::add_u64(base, idx, p);

    // store big-endian 32 bytes at priv32_out[idx*32 ... +31]
    uint8_t* dst = priv32_out + idx*32;
    // write big-endian (mirror of u256_to_bytes_be)
    for (int limb=3; limb>=0; --limb){
        uint64_t w = p.v[limb];
        for (int j=7;j>=0;--j){
            dst[(3-limb)*8 + (7-j)] = (uint8_t)((w >> (j*8)) & 0xFF);
        }
    }
}

// Random within a given base window is handled on host by randomizing base.
// (GPU PRNG for 256-bit uniform in large range is non-trivial and slower due to rejection.)

// ===================== OpenSSL: derive compressed pubkey and 24-byte fingerprint ============

static bool priv_to_pubkey_and_fingerprint(const uint8_t priv32[32], std::array<uint8_t, 33>& out_pubkey, std::array<uint8_t, 24>& out_fingerprint) {
    bool ok=false;
    EC_KEY* key = nullptr;
    EC_GROUP* group = nullptr;
    BIGNUM* bn = nullptr;

    do {
        group = EC_GROUP_new_by_curve_name(NID_secp256k1);
        if (!group) break;

        key = EC_KEY_new();
        if (!key) break;
        if (EC_KEY_set_group(key, group) != 1) break;

        bn = BN_bin2bn(priv32, 32, nullptr);
        if (!bn) break;

        if (EC_KEY_set_private_key(key, bn) != 1) break;

        // pub = priv * G
        EC_POINT* pub = EC_POINT_new(group);
        if (!pub) break;
        if (EC_POINT_mul(group, pub, bn, nullptr, nullptr, nullptr) != 1) {
            EC_POINT_free(pub);
            break;
        }
        if (EC_KEY_set_public_key(key, pub) != 1) {
            EC_POINT_free(pub);
            break;
        }

        // serialize compressed (33 bytes) into out_pubkey
        size_t len = EC_POINT_point2oct(group, pub, POINT_CONVERSION_COMPRESSED, out_pubkey.data(), out_pubkey.size(), nullptr);
        EC_POINT_free(pub);
        if (len != 33) break;

        // fingerprint = last 24 bytes of the X coordinate.
        // The X coordinate is bytes 1-32 of the compressed key.
        // So we want to copy from offset 9 (1 + 8) of the compressed key.
        std::copy_n(out_pubkey.begin() + 9, 24, out_fingerprint.begin());

        ok=true;
    } while(false);

    if (bn) BN_free(bn);
    if (key) EC_KEY_free(key);
    if (group) EC_GROUP_free(group);
    return ok;
}

static inline std::string hex32(const uint8_t* b32){
    std::ostringstream oss;
    oss << std::hex << std::nouppercase << std::setfill('0');
    for (int i=0;i<32;++i) oss << std::setw(2) << (unsigned)b32[i];
    return oss.str();
}
static inline std::string hex_bytes(const uint8_t* data, size_t len){
    std::ostringstream oss;
    oss << std::hex << std::nouppercase << std::setfill('0');
    for (size_t i=0;i<len;++i) oss << std::setw(2) << (unsigned)data[i];
    return oss.str();
}

// ===================== CLI parsing =====================

struct Args {
    U256 start, end;
    std::string target;
    std::string outputFile;
    bool randomMode = false;
    uint64_t count = 1<<20; // default 1M per batch
    int block = 256;
};

static void usage(const char* prog){
    std::cerr <<
      "Usage: " << prog << " --keyspace <hex_start:hex_end> -t <target_or_file> --random [-o <output.bin>] [--count N] [--block B]\n"
      "  --keyspace <start:end> : Required. Hexadecimal range for private keys.\n"
      "  -t <target>            : Required. Target public key (hex) or file with keys.\n"
      "  -R, --random           : Required. Use random generation mode.\n"
      "  -o <file>              : Optional. File to write found private keys and fingerprints.\n"
      "  --count <N>            : Optional. Keys to generate per batch (default: 1048576).\n"
      "  --block <B>            : Optional. CUDA threads per block (default: 256).\n"
      "Example:\n"
      "  " << prog << " --keyspace 4000...:7FFF... -t targets.txt -R -o database.bin\n";
}

static bool parse_args(int argc, char** argv, Args& a){
    bool keyspace_set = false;
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
        } else if (s == "-o" && i+1<argc) {
            a.outputFile = argv[++i];
        } else if (s == "--random" || s == "-R") {
            a.randomMode = true;
        } else if (s=="--count" && i+1<argc){
            a.count = std::strtoull(argv[++i], nullptr, 10);
        } else if (s=="--block" && i+1<argc){
            a.block = std::atoi(argv[++i]);
        } else {
            std::cerr<<"Unknown or malformed arg: "<<s<<"\n"; usage(argv[0]); return false;
        }
    }

    if (!keyspace_set || a.target.empty() || !a.randomMode) {
        std::cerr << "Error: --keyspace, -t, and --random/-R are all required.\n";
        usage(argv[0]);
        return false;
    }
    if (a.start.cmp(a.end) > 0){ std::cerr<<"Error: start of keyspace is greater than end.\n"; return false; }
    if (a.count==0){ std::cerr<<"Error: --count must be > 0.\n"; return false; }
    return true;
}

// ===================== Target Loading =====================

// Helper to convert hex string to bytes
static std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    std::string hex_copy = hex;
    if (hex.length() % 2 != 0) {
        hex_copy = "0" + hex;
    }
    for (unsigned int i = 0; i < hex_copy.length(); i += 2) {
        std::string byteString = hex_copy.substr(i, 2);
        char* end;
        long byte = strtol(byteString.c_str(), &end, 16);
        if (*end != '\0') {
            std::cerr << "Warning: non-hex character encountered in '" << byteString << "'" << std::endl;
            return {};
        }
        bytes.push_back(static_cast<uint8_t>(byte));
    }
    return bytes;
}

// Map from 24-byte fingerprint to 33-byte compressed public key
using TargetMap = std::map<std::array<uint8_t, 24>, std::array<uint8_t, 33>>;

static bool load_targets(const std::string& target_arg, TargetMap& targets) {
    auto process_key = [&](std::string key_hex) {
        // Trim whitespace
        auto first = key_hex.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) return;
        auto last = key_hex.find_last_not_of(" \t\n\r");
        key_hex = key_hex.substr(first, (last - first + 1));

        if (key_hex.length() != 66) {
            std::cerr << "Warning: Invalid public key length (" << key_hex.length() << "), skipping: " << key_hex << std::endl;
            return;
        }
        std::vector<uint8_t> pubkey_bytes = hex_to_bytes(key_hex);
        if (pubkey_bytes.size() != 33) {
             std::cerr << "Warning: Invalid public key bytes after hex conversion, skipping: " << key_hex << std::endl;
            return;
        }

        std::array<uint8_t, 33> pubkey_arr;
        std::copy_n(pubkey_bytes.begin(), 33, pubkey_arr.begin());

        std::array<uint8_t, 24> fingerprint;
        // Compressed pubkey is [1-byte prefix][32-byte X]. Last 24 bytes of X are at offset 9.
        std::copy_n(pubkey_bytes.begin() + 9, 24, fingerprint.begin());

        targets[fingerprint] = pubkey_arr;
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

    std::cout << "Loaded " << targets.size() << " target fingerprints into memory." << std::endl;
    return true;
}


// ===================== Main =====================

int main(int argc, char** argv){
    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    // Load targets
    TargetMap targets;
    if (!load_targets(args.target, targets)) {
        return 1;
    }

    // Open output files
    std::ofstream matches_file("matches.bin", std::ios::binary | std::ios::app);
    if (!matches_file) {
        std::cerr << "Error: Could not open matches.bin for writing." << std::endl;
        return 1;
    }

    std::ofstream output_db_file;
    if (!args.outputFile.empty()) {
        output_db_file.open(args.outputFile, std::ios::binary | std::ios::app);
        if (!output_db_file) {
            std::cerr << "Error: Could not open " << args.outputFile << " for writing." << std::endl;
            return 1;
        }
        std::cout << "Opened database file for writing: " << args.outputFile << std::endl;
    }

    // Device buffers
    uint8_t* d_priv = nullptr;
    cudaMalloc(&d_priv, args.count * 32);
    if (!d_priv){ std::cerr<<"cudaMalloc failed\n"; return 1; }

    // Host buffer
    std::vector<uint8_t> h_priv(args.count * 32);

    // Kernel config
    int block = args.block;
    int grid = (int)((args.count + block - 1) / block);

    // Prepare OpenSSL once to warm up
    {
        uint8_t zero[32] = {0};
        std::array<uint8_t, 33> pub;
        std::array<uint8_t, 24> fp;
        priv_to_pubkey_and_fingerprint(zero, pub, fp);
    }

    uint64_t total_keys_processed = 0;
    auto loop_start_time = std::chrono::high_resolution_clock::now();

    // Main search loop
    while (true) {
        U256 base = random_base_in_range(args.start, args.end, args.count);

        // Fill GPU memory with a batch of private keys
        fill_seq_kernel<<<grid, block>>>(base, d_priv, args.count);

        // Copy private keys from GPU to host. This call is synchronous.
        cudaMemcpy(h_priv.data(), d_priv, args.count * 32, cudaMemcpyDeviceToHost);

        // Process keys on the CPU
        for (uint64_t i = 0; i < args.count; ++i) {
            uint8_t* k = &h_priv[i * 32];

            std::array<uint8_t, 33> pubkey;
            std::array<uint8_t, 24> fingerprint;
            if (priv_to_pubkey_and_fingerprint(k, pubkey, fingerprint)) {
                auto it = targets.find(fingerprint);
                if (it != targets.end()) {
                    // Match found!
                    const auto& matched_pubkey = it->second;

                    // Print to console, ensuring it doesn't get overwritten by status line
                    std::cout << "\n\n!!! MATCH FOUND !!!\n"
                              << "  Private Key: " << hex32(k) << "\n"
                              << "  Public Key:  " << hex_bytes(pubkey.data(), pubkey.size()) << "\n"
                              << "  Target Key:  " << hex_bytes(matched_pubkey.data(), matched_pubkey.size()) << "\n"
                              << std::endl;

                    // Write to matches.bin
                    matches_file.write(reinterpret_cast<const char*>(k), 32);
                    matches_file.write(reinterpret_cast<const char*>(fingerprint.data()), fingerprint.size());
                    matches_file.write(reinterpret_cast<const char*>(matched_pubkey.data()), matched_pubkey.size());
                    matches_file.flush();

                } else {
                    // No match, write to DB if enabled
                    if (output_db_file.is_open()) {
                        output_db_file.write(reinterpret_cast<const char*>(k), 32);
                        output_db_file.write(reinterpret_cast<const char*>(fingerprint.data()), fingerprint.size());
                    }
                }
            }
        }

        total_keys_processed += args.count;
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - loop_start_time).count();

        if (elapsed_seconds >= 2.0) { // Print stats every 2 seconds
            double keys_per_second = total_keys_processed / elapsed_seconds;
            // Use carriage return to print on the same line
            std::cout << "\rSpeed: " << std::fixed << std::setprecision(2) << (keys_per_second / 1e6) << " Mkeys/s  " << std::flush;
            total_keys_processed = 0;
            loop_start_time = current_time;
        }
    }

    // Cleanup
    cudaFree(d_priv);
    return 0;
}
