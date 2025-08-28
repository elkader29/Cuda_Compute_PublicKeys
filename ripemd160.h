#ifndef RIPEMD160_H
#define RIPEMD160_H

/*************************** HEADER FILES ***************************/
#include <cstddef>
#include <cstdint>
#include <memory>

/****************************** MACROS ******************************/
#define RIPEMD160_BLOCK_SIZE 20
#define READ_LE32(p) \
    (((WORD)((p)[0])) | ((WORD)((p)[1]) << 8) | ((WORD)((p)[2]) << 16) | ((WORD)((p)[3]) << 24))
#define WRITE_LE32(p, v) \
    do { \
        (p)[0] = (BYTE)((v)); \
        (p)[1] = (BYTE)((v) >> 8); \
        (p)[2] = (BYTE)((v) >> 16); \
        (p)[3] = (BYTE)((v) >> 24); \
    } while(0)
#define ROL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define F(x, y, z) ((x) ^ (y) ^ (z))
#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define H(x, y, z) (((x) | ~(y)) ^ (z))
#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define J(x, y, z) ((x) ^ ((y) | ~(z)))
#define FF(a, b, c, d, e, x, s) { (a) += F((b), (c), (d)) + (x); (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define GG(a, b, c, d, e, x, s) { (a) += G((b), (c), (d)) + (x) + 0x5a827999; (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define HH(a, b, c, d, e, x, s) { (a) += H((b), (c), (d)) + (x) + 0x6ed9eba1; (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define II(a, b, c, d, e, x, s) { (a) += I((b), (c), (d)) + (x) + 0x8f1bbcdc; (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define JJ(a, b, c, d, e, x, s) { (a) += J((b), (c), (d)) + (x) + 0xa953fd4e; (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define FFF(a, b, c, d, e, x, s) { (a) += F((b), (c), (d)) + (x); (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define GGG(a, b, c, d, e, x, s) { (a) += G((b), (c), (d)) + (x) + 0x7a6d76e9; (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define HHH(a, b, c, d, e, x, s) { (a) += H((b), (c), (d)) + (x) + 0x6d703ef3; (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define III(a, b, c, d, e, x, s) { (a) += I((b), (c), (d)) + (x) + 0x5c4dd124; (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }
#define JJJ(a, b, c, d, e, x, s) { (a) += J((b), (c), (d)) + (x) + 0x50a28be6; (a) = ROL((a), (s)) + (e); (c) = ROL((c), 10); }

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;
typedef unsigned int  WORD;

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

/*********************** FUNCTION DEFINITIONS ***********************/
__device__ inline void RIPEMD160::transform(const unsigned char *message, unsigned int block_nb)
{
	WORD x[16];
	WORD a, b, c, d, e;
	WORD aa, bb, cc, dd, ee;
	for (unsigned int i = 0; i < block_nb; i++) {
		for (unsigned int j = 0; j < 16; j++) { x[j] = READ_LE32(message + (i * 64) + (j * 4)); }
		a = aa = m_h[0]; b = bb = m_h[1]; c = cc = m_h[2]; d = dd = m_h[3]; e = ee = m_h[4];
		FF(a, b, c, d, e, x[0], 11); FF(d, e, a, b, c, x[1], 14); FF(c, d, e, a, b, x[2], 15); FF(b, c, d, e, a, x[3], 12);
		FF(a, b, c, d, e, x[4], 5); FF(d, e, a, b, c, x[5], 8); FF(c, d, e, a, b, x[6], 7); FF(b, c, d, e, a, x[7], 9);
		FF(a, b, c, d, e, x[8], 11); FF(d, e, a, b, c, x[9], 13); FF(c, d, e, a, b, x[10], 14); FF(b, c, d, e, a, x[11], 15);
		FF(a, b, c, d, e, x[12], 6); FF(d, e, a, b, c, x[13], 7); FF(c, d, e, a, b, x[14], 9); FF(b, c, d, e, a, x[15], 8);
		GG(e, a, b, c, d, x[7], 7); GG(d, e, a, b, c, x[4], 6); GG(c, d, e, a, b, x[13], 8); GG(b, c, d, e, a, x[1], 13);
		GG(a, b, c, d, e, x[10], 11); GG(e, a, b, c, d, x[6], 9); GG(d, e, a, b, c, x[15], 7); GG(c, d, e, a, b, x[3], 15);
		GG(b, c, d, e, a, x[12], 7); GG(a, b, c, d, e, x[0], 12); GG(e, a, b, c, d, x[9], 15); GG(d, e, a, b, c, x[5], 9);
		GG(c, d, e, a, b, x[2], 11); GG(b, c, d, e, a, x[14], 7); GG(a, b, c, d, e, x[11], 13); GG(e, a, b, c, d, x[8], 12);
		HH(d, e, a, b, c, x[3], 11); HH(c, d, e, a, b, x[10], 13); HH(b, c, d, e, a, x[14], 6); HH(a, b, c, d, e, x[4], 7);
		HH(e, a, b, c, d, x[9], 14); HH(d, e, a, b, c, x[15], 9); HH(c, d, e, a, b, x[8], 13); HH(b, c, d, e, a, x[1], 15);
		HH(a, b, c, d, e, x[2], 14); HH(e, a, b, c, d, x[7], 8); HH(d, e, a, b, c, x[0], 13); HH(c, d, e, a, b, x[6], 6);
		HH(b, c, d, e, a, x[13], 5); HH(a, b, c, d, e, x[11], 12); HH(e, a, b, c, d, x[5], 7); HH(d, e, a, b, c, x[12], 5);
		II(c, d, e, a, b, x[1], 11); II(b, c, d, e, a, x[9], 12); II(a, b, c, d, e, x[11], 14); II(e, a, b, c, d, x[10], 15);
		II(d, e, a, b, c, x[0], 14); II(c, d, e, a, b, x[8], 15); II(b, c, d, e, a, x[12], 9); II(a, b, c, d, e, x[4], 8);
		II(e, a, b, c, d, x[13], 9); II(d, e, a, b, c, x[5], 14); II(c, d, e, a, b, x[2], 8); II(b, c, d, e, a, x[14], 8);
		II(a, b, c, d, e, x[7], 6); II(e, a, b, c, d, x[6], 6); II(d, e, a, b, c, x[15], 5); II(c, d, e, a, b, x[3], 12);
		JJ(b, c, d, e, a, x[4], 9); JJ(a, b, c, d, e, x[0], 15); JJ(e, a, b, c, d, x[5], 5); JJ(d, e, a, b, c, x[9], 11);
		JJ(c, d, e, a, b, x[7], 6); JJ(b, c, d, e, a, x[12], 8); JJ(a, b, c, d, e, x[2], 13); JJ(e, a, b, c, d, x[10], 12);
		JJ(d, e, a, b, c, x[14], 5); JJ(c, d, e, a, b, x[1], 12); JJ(b, c, d, e, a, x[3], 13); JJ(a, b, c, d, e, x[8], 14);
		JJ(e, a, b, c, d, x[11], 11); JJ(d, e, a, b, c, x[6], 8); JJ(c, d, e, a, b, x[15], 5); JJ(b, c, d, e, a, x[13], 6);
		JJJ(aa, bb, cc, dd, ee, x[5], 8); JJJ(dd, ee, aa, bb, cc, x[14], 9); JJJ(cc, dd, ee, aa, bb, x[7], 9); JJJ(bb, cc, dd, ee, aa, x[0], 11);
		JJJ(aa, bb, cc, dd, ee, x[9], 13); JJJ(dd, ee, aa, bb, cc, x[2], 15); JJJ(cc, dd, ee, aa, bb, x[11], 15); JJJ(bb, cc, dd, ee, aa, x[4], 5);
		JJJ(aa, bb, cc, dd, ee, x[13], 7); JJJ(dd, ee, aa, bb, cc, x[6], 7); JJJ(cc, dd, ee, aa, bb, x[15], 8); JJJ(bb, cc, dd, ee, aa, x[8], 11);
		JJJ(aa, bb, cc, dd, ee, x[1], 14); JJJ(dd, ee, aa, bb, cc, x[10], 14); JJJ(cc, dd, ee, aa, bb, x[3], 12); JJJ(bb, cc, dd, ee, aa, x[12], 6);
		III(ee, aa, bb, cc, dd, x[6], 9); III(dd, ee, aa, bb, cc, x[11], 13); III(cc, dd, ee, aa, bb, x[3], 15); III(bb, cc, dd, ee, aa, x[7], 7);
		III(aa, bb, cc, dd, ee, x[0], 12); III(ee, aa, bb, cc, dd, x[13], 5); III(dd, ee, aa, bb, cc, x[5], 9); III(cc, dd, ee, aa, bb, x[10], 11);
		III(bb, cc, dd, ee, aa, x[14], 7); III(aa, bb, cc, dd, ee, x[15], 7); III(ee, aa, bb, cc, dd, x[8], 12); III(dd, ee, aa, bb, cc, x[12], 7);
		III(cc, dd, ee, aa, bb, x[4], 6); III(bb, cc, dd, ee, aa, x[9], 15); III(aa, bb, cc, dd, ee, x[1], 13); III(ee, aa, bb, cc, dd, x[2], 11);
		HHH(dd, ee, aa, bb, cc, x[15], 9); HHH(cc, dd, ee, aa, bb, x[5], 7); HHH(bb, cc, dd, ee, aa, x[1], 15); HHH(aa, bb, cc, dd, ee, x[3], 11);
		HHH(ee, aa, bb, cc, dd, x[7], 8); HHH(dd, ee, aa, bb, cc, x[14], 6); HHH(cc, dd, ee, aa, bb, x[6], 6); HHH(bb, cc, dd, ee, aa, x[9], 14);
		HHH(aa, bb, cc, dd, ee, x[11], 12); HHH(ee, aa, bb, cc, dd, x[0], 13); HHH(dd, ee, aa, bb, cc, x[4], 7); HHH(cc, dd, ee, aa, bb, x[10], 13);
		HHH(bb, cc, dd, ee, aa, x[13], 5); HHH(aa, bb, cc, dd, ee, x[2], 14); HHH(ee, aa, bb, cc, dd, x[12], 5); HHH(dd, ee, aa, bb, cc, x[8], 13);
		GGG(cc, dd, ee, aa, bb, x[8], 15); GGG(bb, cc, dd, ee, aa, x[6], 5); GGG(aa, bb, cc, dd, ee, x[4], 8); GGG(ee, aa, bb, cc, dd, x[1], 11);
		GGG(dd, ee, aa, bb, cc, x[3], 14); GGG(cc, dd, ee, aa, bb, x[11], 14); GGG(bb, cc, dd, ee, aa, x[15], 6); GGG(aa, bb, cc, dd, ee, x[0], 14);
		GGG(ee, aa, bb, cc, dd, x[5], 6); GGG(dd, ee, aa, bb, cc, x[12], 9); GGG(cc, dd, ee, aa, bb, x[2], 12); GGG(bb, cc, dd, ee, aa, x[13], 9);
		GGG(aa, bb, cc, dd, ee, x[9], 12); GGG(ee, aa, bb, cc, dd, x[7], 5); GGG(dd, ee, aa, bb, cc, x[10], 15); GGG(cc, dd, ee, aa, bb, x[14], 8);
		FFF(bb, cc, dd, ee, aa, x[12], 8); FFF(aa, bb, cc, dd, ee, x[0], 15); FFF(ee, aa, bb, cc, dd, x[10], 12); FFF(dd, ee, aa, bb, cc, x[4], 9);
		FFF(cc, dd, ee, aa, bb, x[1], 12); FFF(bb, cc, dd, ee, aa, x[5], 5); FFF(aa, bb, cc, dd, ee, x[8], 14); FFF(ee, aa, bb, cc, dd, x[7], 6);
		FFF(dd, ee, aa, bb, cc, x[6], 8); FFF(cc, dd, ee, aa, bb, x[2], 13); FFF(bb, cc, dd, ee, aa, x[13], 6); FFF(aa, bb, cc, dd, ee, x[14], 5);
		FFF(ee, aa, bb, cc, dd, x[3], 13); FFF(dd, ee, aa, bb, cc, x[11], 11); FFF(cc, dd, ee, aa, bb, x[0], 15); FFF(bb, cc, dd, ee, aa, x[9], 11);
		dd += c + m_h[1]; m_h[1] = m_h[2] + d + ee; m_h[2] = m_h[3] + e + aa; m_h[3] = m_h[4] + a + bb; m_h[4] = m_h[0] + b + cc; m_h[0] = dd;
	}
}

__device__ inline RIPEMD160::RIPEMD160()
{
	m_h[0] = 0x67452301;
	m_h[1] = 0xefcdab89;
	m_h[2] = 0x98badcfe;
	m_h[3] = 0x10325476;
	m_h[4] = 0xc3d2e1f0;
	m_len = 0;
	m_tot_len = 0;
}

__device__ inline void RIPEMD160::update(const unsigned char *message, unsigned int len)
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

__device__ inline void RIPEMD160::final(unsigned char *digest)
{
	unsigned int block_nb;
	unsigned int pm_len;
	block_nb = (1 + ((64 - 9) < (m_len % 64)));
	pm_len = block_nb << 6;
	for(unsigned int idx = m_len; idx < pm_len; ++idx)
        m_block[idx] = 0;
	m_block[m_len] = 0x80;
	WRITE_LE32(m_block + pm_len - 8, (m_tot_len + m_len) << 3);
    WRITE_LE32(m_block + pm_len - 4, 0);
	transform(m_block, block_nb);
	for (int i = 0; i < 5; i++) {
		WRITE_LE32(digest + i * 4, m_h[i]);
	}
}

__device__ inline void RIPEMD160::hash(const unsigned char* data, size_t len, unsigned char* digest)
{
    RIPEMD160 ctx;
    ctx.update(data, len);
    ctx.final(digest);
}

#endif
