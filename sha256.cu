/*********************************************************************
* Filename:   sha256.cu
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the SHA-256 algorithm.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include "sha256.h"
#include <memory>

/****************************** MACROS ******************************/
#define SHA256_READ_UNALIGNED_BIG_ENDIAN(p) \
    ( ((WORD)((p)[0]) << 24) | ((WORD)((p)[1]) << 16) | ((WORD)((p)[2]) << 8) | ((WORD)((p)[3])) )

#define SHA256_WRITE_UNALIGNED_BIG_ENDIAN(p, v) \
    do { \
        (p)[0] = (BYTE)((v) >> 24); \
        (p)[1] = (BYTE)((v) >> 16); \
        (p)[2] = (BYTE)((v) >> 8); \
        (p)[3] = (BYTE)((v)); \
    } while(0)

#define ROTLEFT(a,b)  (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)     (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x)     (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x)    (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x)    (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
__constant__ static const WORD k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** FUNCTION DEFINITIONS ***********************/
__device__ void SHA256::transform(const unsigned char *message, unsigned int block_nb)
{
	WORD w[64], wv[8], t1, t2;
	const unsigned char *sub_block;
	int i;

	for (i = 0; i < (int) block_nb; i++) {
		sub_block = message + (i << 6);

		for (int j = 0; j < 16; j++) {
			w[j] = SHA256_READ_UNALIGNED_BIG_ENDIAN(sub_block + (j << 2));
		}

		for (int j = 16; j < 64; j++) {
			w[j] = SIG1(w[j - 2]) + w[j - 7] + SIG0(w[j - 15]) + w[j - 16];
		}

		for (int j = 0; j < 8; j++) {
			wv[j] = m_h[j];
		}

		for (int j = 0; j < 64; j++) {
			t1 = wv[7] + EP1(wv[4]) + CH(wv[4], wv[5], wv[6]) + k[j] + w[j];
			t2 = EP0(wv[0]) + MAJ(wv[0], wv[1], wv[2]);
			wv[7] = wv[6];
			wv[6] = wv[5];
			wv[5] = wv[4];
			wv[4] = wv[3] + t1;
			wv[3] = wv[2];
			wv[2] = wv[1];
			wv[1] = wv[0];
			wv[0] = t1 + t2;
		}

		for (int j = 0; j < 8; j++) {
			m_h[j] += wv[j];
		}
	}
}

__device__ SHA256::SHA256()
{
	m_h[0] = 0x6a09e667;
	m_h[1] = 0xbb67ae85;
	m_h[2] = 0x3c6ef372;
	m_h[3] = 0xa54ff53a;
	m_h[4] = 0x510e527f;
	m_h[5] = 0x9b05688c;
	m_h[6] = 0x1f83d9ab;
	m_h[7] = 0x5be0cd19;
	m_len = 0;
	m_tot_len = 0;
}

__device__ void SHA256::update(const unsigned char *message, unsigned int len)
{
	unsigned int block_nb;
	unsigned int new_len, rem_len, tmp_len;
	const unsigned char *shifted_message;

	tmp_len = 64 - m_len;
	rem_len = len < tmp_len ? len : tmp_len;

    for(unsigned int i = 0; i < rem_len; ++i)
        m_block[m_len + i] = message[i];

	if (m_len + len < 64) {
		m_len += len;
		return;
	}

	new_len = len - rem_len;
	block_nb = new_len / 64;
	shifted_message = message + rem_len;
	transform(m_block, 1);
	transform(shifted_message, block_nb);
	rem_len = new_len % 64;

    for(unsigned int i = 0; i < rem_len; ++i)
        m_block[i] = shifted_message[block_nb << 6 + i];

    m_len = rem_len;
	m_tot_len += (block_nb + 1) << 6;
}

__device__ void SHA256::final(unsigned char *digest)
{
	unsigned int block_nb;
	unsigned int pm_len;
	unsigned int len_b;
	int i;

	block_nb = (1 + ((64 - 9) < (m_len % 64)));
	len_b = (m_tot_len + m_len) << 3;
	pm_len = block_nb << 6;

	for(unsigned int idx = m_len; idx < pm_len; ++idx)
        m_block[idx] = 0;
	m_block[m_len] = 0x80;

    SHA256_WRITE_UNALIGNED_BIG_ENDIAN(m_block + pm_len - 4, len_b);

	transform(m_block, block_nb);

	for (i = 0; i < 8; i++) {
		SHA256_WRITE_UNALIGNED_BIG_ENDIAN(digest + i * 4, m_h[i]);
	}
}

// Wrapper functions for C-style interface
__device__ void sha256_init(SHA256* ctx) {
    new (ctx) SHA256();
}

__device__ void sha256_update(SHA256* ctx, const BYTE* data, size_t len) {
    ctx->update(data, len);
}

__device__ void sha256_final(SHA256* ctx, BYTE* hash) {
    ctx->final(hash);
}
