/*********************************************************************
* Filename:   sha256.h
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Defines the API for the corresponding SHA1 implementation.
*********************************************************************/

#ifndef SHA256_H
#define SHA256_H

/*************************** HEADER FILES ***************************/
#include <cstddef>
#include <cstdint>

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines

class SHA256
{
protected:
	__device__ void transform(const unsigned char *message, unsigned int block_nb);
	unsigned int m_tot_len;
	unsigned int m_len;
	unsigned char m_block[2 * 64];
	WORD m_h[8];

public:
	__device__ SHA256();
	__device__ void update(const unsigned char *message, unsigned int len);
	__device__ void final(unsigned char *digest);
	__device__ static void hash(const unsigned char* data, size_t len, unsigned char* digest);
};

// Functions for CUDA
__device__ void sha256_init(SHA256* ctx);
__device__ void sha256_update(SHA256* ctx, const BYTE* data, size_t len);
__device__ void sha256_final(SHA256* ctx, BYTE* hash);

// Must be in header for separate compilation to work
__device__ inline void SHA256::hash(const unsigned char* data, size_t len, unsigned char* digest)
{
    SHA256 ctx;
    new (&ctx) SHA256();
    ctx.update(data, len);
    ctx.final(digest);
}

#endif   // SHA256_H
