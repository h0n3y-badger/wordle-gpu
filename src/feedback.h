// Wordle feedback, shared by host and device code.
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define WG_HD __host__ __device__ __forceinline__
#else
#define WG_HD inline
#endif

constexpr int NPAT = 243;          // 3^5 feedback patterns
constexpr uint8_t ALL_GREEN = 242; // 22222 in base 3

// Digit i (base 3, position 0 lowest) is 0 gray, 1 yellow, 2 green.
// Repeated letters follow the game: greens are matched first, then earlier positions
// take the remaining copies of a letter before later ones.
WG_HD uint8_t feedback(const char * g, const char * a)
{
	bool green[5];
	for (int i = 0; i < 5; i++) green[i] = g[i] == a[i];
	int pat = 0;
	for (int i = 4; i >= 0; i--)
	{
		int d = 2;
		if (!green[i])
		{
			int avail = 0, before = 0;
			for (int k = 0; k < 5; k++) avail += !green[k] && a[k] == g[i];
			for (int j = 0; j < i; j++) before += !green[j] && g[j] == g[i];
			d = before < avail;
		}
		pat = pat * 3 + d;
	}
	return (uint8_t)pat;
}

WG_HD int pat_digit(int pat, int i)
{
	for (; i > 0; i--) pat /= 3;
	return pat % 3;
}

// Hard mode: greens must stay in place and every revealed letter must be reused
// (as many copies as were shown green or yellow in one guess).
struct HardRule
{
	char green[5] = {0, 0, 0, 0, 0}; // 0 = free
	char letter[5] = {0, 0, 0, 0, 0}; // required letters, 0-terminated if fewer than 5
	uint8_t count[5] = {0, 0, 0, 0, 0};

	WG_HD void add(const char * g, int pat)
	{
		for (int i = 0; i < 5; i++)
			if (pat_digit(pat, i) == 2) green[i] = g[i];
		for (int i = 0; i < 5; i++)
		{
			if (pat_digit(pat, i) == 0) continue;
			int n = 0;
			for (int j = 0; j < 5; j++) n += g[j] == g[i] && pat_digit(pat, j) != 0;
			int k = 0;
			while (k < 5 && letter[k] && letter[k] != g[i]) k++;
			if (k == 5) continue; // cannot happen: at most 5 distinct letters are shown
			letter[k] = g[i];
			if (count[k] < n) count[k] = (uint8_t)n;
		}
	}

	WG_HD bool allows(const char * w) const
	{
		for (int i = 0; i < 5; i++)
			if (green[i] && w[i] != green[i]) return false;
		for (int k = 0; k < 5 && letter[k]; k++)
		{
			int n = 0;
			for (int j = 0; j < 5; j++) n += w[j] == letter[k];
			if (n < count[k]) return false;
		}
		return true;
	}
};
