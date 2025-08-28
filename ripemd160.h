/*********************************************************************
* Filename:   ripemd160.h
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Defines the API for the corresponding RIPEMD-160 implementation.
*********************************************************************/

#ifndef RIPEMD160_H
#define RIPEMD160_H

/*************************** HEADER FILES ***************************/
#include <cstddef>
#include <cstdint>

/****************************** MACROS ******************************/
#define RIPEMD160_BLOCK_SIZE 20            // RIPEMD-160 outputs a 20 byte digest

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines

class RIPEMD160
{
protected:
	__device__ void transform(const unsigned char *message, unsigned int block_nb);
	unsigned int m_tot_len;
	unsigned int m_len;
	unsigned char m_block[64];
	WORD m_h[5];

public:
	__device__ RIPEMD160();
	__device__ void update(const unsigned char *message, unsigned int len);
	__device__ void final(unsigned char *digest);
	__device__ static void hash(const unsigned char* data, size_t len, unsigned char* digest);
};

// Must be in header for separate compilation to work
__device__ inline void RIPEMD160::hash(const unsigned char* data, size_t len, unsigned char* digest)
{
    RIPEMD160 ctx;
    new (&ctx) RIPEMD160();
    ctx.update(data, len);
    ctx.final(digest);
}

#endif   // RIPEMD160_H
