# Cuda Public Key Search

A high-performance tool using CUDA to search for secp256k1 public keys.
It generates private keys in a given range, derives compressed public keys,
and compares their 24-byte fingerprints against a list of targets.

## Build

The tool requires OpenSSL for cryptographic operations. Ensure you have OpenSSL development libraries installed.

Build the tool using `nvcc`:
```sh
nvcc -O3 -std=c++17 gpu_search.cu -lssl -lcrypto -o gpu_search
```

## Usage

The tool operates in a random search mode within a specified hexadecimal range.

```
Usage: ./gpu_search --keyspace <hex_start:hex_end> -t <target_or_file> --random [-o <output.bin>] [--count N] [--block B]
```

### Options

- `--keyspace <start:end>`: **Required.** Hexadecimal range for private keys.
- `-t <target>`: **Required.** A single target public key (compressed, 66 hex chars) or a file containing one public key per line.
- `-R`, `--random`: **Required.** Use random generation mode.
- `-o <file>`: Optional. File to write the database of generated private keys and their public key fingerprints. The format for each entry is `[32-byte private key][24-byte fingerprint]`.
- `--count <N>`: Optional. Number of keys to generate per batch (default: 1048576).
- `--block <B>`: Optional. CUDA threads per block (default: 256).

### Example

Search for public keys listed in `targets.txt` within the specified range and save all generated attempts to `database.bin`:

```sh
./gpu_search --keyspace 4000000000000000000000000000000000:7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF -t targets.txt -R -o database.bin
```

If a match is found, it will be printed to the console and saved to `matches.bin`.
