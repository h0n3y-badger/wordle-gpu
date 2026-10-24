#include "exact.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#if defined(__AVX512BW__) || defined(__BMI2__)
#include <immintrin.h>
#endif

namespace {

struct Cancelled {};

// Lower bounds from how finely any guess can split a set. Say every candidate guess gives
// at most km distinct feedbacks on a set U and every other allowed guess at most kn. Below
// U (subsets of it, under hard-mode rules that only get stricter) a candidate guess still
// gives at most km, and any other guess at most max(kn, km - 1): a word of U that is no
// longer a candidate has lost its own all-green class. No strategy for U beats the
// cheapest tree obeying those caps, and with h guesses left there may be none (INF).
// It bites in hard mode: once a family like _ASTE or _IGHT has its letters fixed, its
// words can only be tried one at a time, t(t+1)/2 guesses in total and t deep.
struct CapBound
{
	static constexpr int T = 32, H = 6; // set sizes and guesses left covered
	// root: U itself, t words, h guesses left; sub: any subset of U of t words.
	std::vector<int> root, sub;
	static size_t at(int h, int km, int kn, int t) { return (((size_t)h * (T + 1) + km) * (T + 1) + kn) * (T + 1) + t; }

	CapBound() : root(at(H + 1, 0, 0, 0), INF), sub(at(H + 1, 0, 0, 0), INF)
	{
		constexpr int N = T + 1;
		// f[h][t]: least total for t words with h guesses left, every node within the caps;
		// q[h][x][k]: least sum of f[h] over at most k parts of x words.
		std::vector<int> f((H + 1) * N), q((size_t)(H + 1) * N * N);
		auto F = [&](int h, int t) { return f[h * N + t]; };
		auto Q = [&](int h, int x, int k) {
			if (x == 0) return 0;
			if (h <= 0 || k <= 0) return INF;
			return q[((size_t)h * N + x) * N + std::min(k, T)];
		};
		for (int km = 1; km <= T; km++)
			for (int kn = 0; kn <= T; kn++)
			{
				const int cn = std::max(kn, km - 1);
				// t words, h left: guess a candidate (it is done, the rest in at most km-1
				// parts) or another word (2 to nmax parts).
				auto node = [&](int h, int t, int nmax) {
					if (t == 0) return 0;
					if (h == 0) return INF;
					int best = Q(h - 1, t - 1, km - 1);
					for (int a = 1; a < t && nmax >= 2; a++) best = std::min(best, F(h - 1, a) + Q(h - 1, t - a, nmax - 1));
					return std::min(INF, t + best);
				};
				for (int h = 0; h <= H; h++)
					for (int t = 0; t <= T; t++)
					{
						f[h * N + t] = node(h, t, cn);
						root[at(h, km, kn, t)] = node(h, t, kn);
						sub[at(h, km, kn, t)] = f[h * N + t];
						if (h == 0 || t == 0) continue;
						int * qt = &q[((size_t)h * N + t) * N];
						qt[0] = INF;
						for (int k = 1; k <= T; k++)
						{
							// Singletons (1 each) are the cheapest split once they fit: f(a) >= 2a-1.
							int v = k >= t ? t : qt[k - 1];
							for (int a = 1; a <= t && k < t; a++) v = std::min(v, F(h, a) + Q(h, t - a, k - 1));
							qt[k] = std::min(v, INF);
						}
					}
			}
	}
};

const CapBound & cap_bound()
{
	static const CapBound c;
	return c;
}

// Pattern counts without clearing all 243 slots per guess.
struct Hist
{
	int cnt[NPAT];
	uint32_t gen[NPAT] = {};
	uint32_t cur = 0;
	uint8_t used[NPAT];
	int nused;

	void reset() { cur++; nused = 0; }
	void add(uint8_t f)
	{
		if (gen[f] != cur) { gen[f] = cur; cnt[f] = 0; used[nused++] = f; }
		cnt[f]++;
	}
	int get(uint8_t f) const { return gen[f] == cur ? cnt[f] : 0; }
};

thread_local Hist tl_hist;

// Sets up to this size count distinct feedbacks 64 guesses at a time in vector registers
// (counts are bytes), padded up to NET_ROWS rows.
constexpr int VEC_MAX = 248;
constexpr int NET_ROWS = 256;

// A node's guess scan, kept while its options are tried. Per chunk of 64 lanes (a block of
// guesses, or in hard mode a run of blocks with only the allowed guesses, packed): each
// lane's distinct feedbacks, the lanes whose bound is below beta, and which are candidates.
struct alignas(64) Lanes { uint8_t v[64]; };
struct Frame
{
	std::vector<Lanes> cnt;
	std::vector<uint64_t> sel, cand;
	std::vector<uint16_t> ids; // the guess in each lane, when packed
	int nch = 0;
	bool packed = false;
	int top[2]; // largest count among the selected non-candidates / candidates, -1 if none
	int most[2]; // largest count among all allowed non-candidates / candidates (left >= 3)

	int guess(int k, int l) const { return packed ? ids[k * 64 + l] : k * 64 + l; }
};

// Frames by recursion depth on this thread: a node's frame outlives the scans of its children.
thread_local std::vector<std::unique_ptr<Frame>> tl_frames;
thread_local int tl_depth = 0;

struct FrameRef
{
	Frame & f;
	explicit FrameRef(int W) : f(get(W)) {}
	~FrameRef() { tl_depth--; }
	FrameRef(const FrameRef &) = delete;
	static Frame & get(int W)
	{
		if ((int)tl_frames.size() <= tl_depth) tl_frames.emplace_back(new Frame);
		Frame & f = *tl_frames[tl_depth++];
		if ((int)f.sel.size() < W)
		{
			f.cnt.resize(W);
			f.sel.resize(W);
			f.cand.resize(W);
		}
		return f;
	}
};

// Per-thread scratch for the guess scan; nothing in it lives across a recursive call.
struct Scratch
{
#ifdef __AVX512BW__
	__m512i stage[VEC_MAX]; // packed rows of one chunk
#endif
	const uint8_t * rows[NET_ROWS];
	std::vector<uint64_t> allow, cand; // one bit per guess
};

thread_local Scratch tl_scratch;

// The constraints in a canonical order, so equal rules reached in a different order share entries.
HardRule canonical(const HardRule & r)
{
	HardRule c = r;
	int m = 0;
	while (m < 5 && c.letter[m]) m++;
	for (int i = 1; i < m; i++)
		for (int j = i; j > 0 && (unsigned char)c.letter[j - 1] > (unsigned char)c.letter[j]; j--)
		{
			std::swap(c.letter[j - 1], c.letter[j]);
			std::swap(c.count[j - 1], c.count[j]);
		}
	return c;
}

// Nodes with at least this many candidates share their options with idle threads.
constexpr int SHARE_MIN = 24;

std::string make_key(const uint16_t * s, int n, int left, const HardRule * rule)
{
	std::string k((const char *)s, n * sizeof(uint16_t));
	k.push_back((char)left);
	if (rule)
	{
		const HardRule c = canonical(*rule);
		k.append((const char *)&c, sizeof(HardRule));
	}
	return k;
}

// Whether some candidate in s tells all n apart, so guessing it costs 2n-1 in total.
// Stops at a candidate's first repeated feedback.
inline bool perfect_split(const Engine & eng, const uint16_t * s, int n)
{
	if (n > NPAT) return false;
	const auto & pool = eng.pool();
	for (int i = 0; i < n; i++)
	{
		const uint8_t * row = eng.row(pool[s[i]]);
		uint64_t seen[4] = { 0, 0, 0, 0 };
		int j = 0;
		for (; j < n; j++)
		{
			const uint8_t f = row[s[j]];
			const uint64_t bit = 1ull << (f & 63);
			if (seen[f >> 6] & bit) break;
			seen[f >> 6] |= bit;
		}
		if (j == n) return true;
	}
	return false;
}

#ifdef __AVX512BW__
// Distinct feedbacks over n candidate rows for 64 neighbouring guesses, one byte per guess.
// Rows past n point at row 0, which never adds a feedback, so a counter may round n up.

// Sorting networks: Batcher's odd-even merge sort for the next power of two, comparators that
// touch a position >= N dropped (those would only ever hold +infinity). Up to 32 rows the
// comparators go layer by layer; above that depth-first (sort each half, then merge), which
// keeps fewer rows live at once and so spills far less.
template <class F>
constexpr void oem_merge(int lo, int n, int r, int N, F & f)
{
	const int m = 2 * r;
	if (m < n)
	{
		oem_merge(lo, n, m, N, f);
		oem_merge(lo + r, n, m, N, f);
		for (int i = lo + r; i + r < lo + n; i += m)
			if (i + r < N) f(i, i + r);
	}
	else if (lo + r < N) f(lo, lo + r);
}
template <class F>
constexpr void oem_sort(int lo, int n, int N, F & f)
{
	if (n < 2) return;
	oem_sort(lo, n / 2, N, f);
	oem_sort(lo + n / 2, n / 2, N, f);
	oem_merge(lo, n, 1, N, f);
}
template <class F>
constexpr void network(int N, F & f)
{
	int P = 1;
	while (P < N) P *= 2;
	if (N > 32)
	{
		oem_sort(0, P, N, f);
		return;
	}
	for (int p = 1; p < P; p += p)
		for (int k = p; k > 0; k /= 2)
			for (int j = k % p; j + k < P; j += 2 * k)
				for (int i = 0; i < k && i + j + k < N; i++)
					if ((i + j) / (2 * p) == (i + j + k) / (2 * p)) f(i + j, i + j + k);
}
template <int N>
struct Network
{
	struct Count { int m = 0; constexpr void operator()(int, int) { m++; } };
	static constexpr int M = [] { Count c; network(N, c); return c.m; }();
	struct Fill { std::array<std::pair<int, int>, M> p{}; int m = 0; constexpr void operator()(int a, int b) { p[m++] = { a, b }; } };
	static constexpr auto pairs = [] { Fill f; network(N, f); return f.p; }();
};

// Sorts the N feedbacks of each guess with a min/max network unrolled at compile time (the
// rows stay in registers, mostly) and counts the changes between neighbours.
template <int N>
__m512i distinct_sorted(const uint8_t * const * rows, size_t off, int)
{
	__m512i v[N];
#pragma GCC unroll 4096
	for (int j = 0; j < N; j++) v[j] = _mm512_load_si512(rows[j] + off);
#pragma GCC unroll 4096
	for (int i = 0; i < Network<N>::M; i++)
	{
		const int a = Network<N>::pairs[i].first, b = Network<N>::pairs[i].second;
		const __m512i x = v[a], y = v[b];
		v[a] = _mm512_min_epu8(x, y);
		v[b] = _mm512_max_epu8(x, y);
	}
	const __m512i one = _mm512_set1_epi8(1);
	__m512i cnt = one;
#pragma GCC unroll 4096
	for (int j = 1; j < N; j++) cnt = _mm512_mask_add_epi8(cnt, _mm512_cmpneq_epu8_mask(v[j], v[j - 1]), cnt, one);
	return cnt;
}

// Network sizes past 32; n rounds up to the next.
#ifndef NET_PAD_LIST
#define NET_PAD_LIST 36, 40, 44, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256
#endif
constexpr int NET_PAD[] = { NET_PAD_LIST };
constexpr int NET_MAX = NET_PAD[sizeof NET_PAD / sizeof NET_PAD[0] - 1];
constexpr int net_size(int n)
{
	if (n <= 32) return n < 2 ? 2 : n;
	if (n > NET_MAX) return (n + 3) & ~3;
	int i = 0;
	while (NET_PAD[i] < n) i++;
	return NET_PAD[i];
}
static_assert(net_size(VEC_MAX) <= NET_ROWS);

// Past the networks: a row adds one wherever it differs from every earlier row. Differences
// are tracked as min(row ^ earlier) per byte (zero where equal), four rows at a time.
inline __m512i distinct_pairs(const uint8_t * const * rows, size_t off, int n)
{
	static thread_local __m512i buf[NET_ROWS];
	const __m512i one = _mm512_set1_epi8(1);
	__m512i cnt = _mm512_setzero_si512();
	for (int j = 0; j < n; j += 4)
	{
		const __m512i a = _mm512_load_si512(rows[j] + off), b = _mm512_load_si512(rows[j + 1] + off);
		const __m512i c = _mm512_load_si512(rows[j + 2] + off), d = _mm512_load_si512(rows[j + 3] + off);
		__m512i da = _mm512_set1_epi8(-1), db = _mm512_xor_si512(b, a);
		__m512i dc = _mm512_min_epu8(_mm512_xor_si512(c, a), _mm512_xor_si512(c, b));
		__m512i dd = _mm512_min_epu8(_mm512_min_epu8(_mm512_xor_si512(d, a), _mm512_xor_si512(d, b)), _mm512_xor_si512(d, c));
		for (int i = 0; i < j; i++)
		{
			const __m512i x = buf[i];
			da = _mm512_min_epu8(da, _mm512_xor_si512(a, x));
			db = _mm512_min_epu8(db, _mm512_xor_si512(b, x));
			dc = _mm512_min_epu8(dc, _mm512_xor_si512(c, x));
			dd = _mm512_min_epu8(dd, _mm512_xor_si512(d, x));
		}
		buf[j] = a;
		buf[j + 1] = b;
		buf[j + 2] = c;
		buf[j + 3] = d;
		cnt = _mm512_add_epi8(cnt, _mm512_min_epu8(da, one));
		cnt = _mm512_add_epi8(cnt, _mm512_min_epu8(db, one));
		cnt = _mm512_add_epi8(cnt, _mm512_min_epu8(dc, one));
		cnt = _mm512_add_epi8(cnt, _mm512_min_epu8(dd, one));
	}
	return cnt;
}

using DistinctFn = __m512i (*)(const uint8_t * const * rows, size_t off, int n);
struct NetTable { DistinctFn fn[VEC_MAX + 1]; };
template <size_t... K>
constexpr NetTable make_nets(std::index_sequence<K...>)
{
	return { { (K <= NET_MAX ? &distinct_sorted<net_size(K <= NET_MAX ? (int)K : 2)> : &distinct_pairs)... } };
}
constexpr NetTable NETS = make_nets(std::make_index_sequence<VEC_MAX + 1>());

inline int max_byte(__m512i v)
{
	const __m256i a = _mm256_max_epu8(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
	__m128i b = _mm_max_epu8(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
	b = _mm_max_epu8(b, _mm_srli_si128(b, 8));
	b = _mm_max_epu8(b, _mm_srli_si128(b, 4));
	b = _mm_max_epu8(b, _mm_srli_si128(b, 2));
	b = _mm_max_epu8(b, _mm_srli_si128(b, 1));
	return _mm_cvtsi128_si32(b) & 255;
}
#endif

// What the guess scan of a node needs.
struct ScanIn
{
	const Engine & eng;
	const uint8_t * pm; // pool-major table
	size_t stride;
	int W;
	const uint16_t * s;
	int n, left, beta, lo;
	const uint64_t * allow, * cand;
	bool dense; // every guess allowed
};

// Scans every allowed guess for its bound n + 2(n-in) - groups = 3n - in - distinct into f.
// Returns true when some guess settles the node at 2n: a non-candidate splitting all apart,
// or (with a guess to spare) a candidate leaving one pair. With two guesses left only that
// is looked for. Otherwise (left >= 3) f selects the guesses with distinct > lo and bound
// < beta, and has the most distinct feedbacks of any allowed guess.
bool scan(const ScanIn & in, Frame & f)
{
	const int n = in.n, left = in.left, lo = in.lo, W = in.W;
	// Non-candidates have bound 3n - distinct.
	const int thr = std::max(lo, 3 * n - in.beta);
	f.packed = false;
	f.nch = W;
#ifdef __AVX512BW__
	if (n <= VEC_MAX)
	{
		Scratch & sc = tl_scratch;
		const DistinctFn df = NETS.fn[n];
		const int padn = net_size(n);
		const __m512i vn = _mm512_set1_epi8((char)n), vn1 = _mm512_set1_epi8((char)(n - 1));
		const __m512i vlo = _mm512_set1_epi8((char)lo), vthr = _mm512_set1_epi8((char)thr);
		__m512i top0 = _mm512_setzero_si512(), top1 = _mm512_setzero_si512();
		__m512i most0 = _mm512_setzero_si512(), most1 = _mm512_setzero_si512();
		// Chunk k's counts for the lanes in valid; true if settled.
		auto finish = [&](int k, __m512i cnt, uint64_t valid, uint64_t cd) {
			const uint64_t done = (_mm512_cmpeq_epi8_mask(cnt, vn) & ~cd) | (left >= 3 ? _mm512_cmpeq_epi8_mask(cnt, vn1) & cd : 0);
			if (done & valid) return true;
			uint64_t sel = 0;
			if (left >= 3)
			{
				sel = ((_mm512_cmpgt_epu8_mask(cnt, vthr) & ~cd) | (_mm512_cmpgt_epu8_mask(cnt, vlo) & cd)) & valid;
				_mm512_store_si512(f.cnt[k].v, cnt);
				top0 = _mm512_mask_max_epu8(top0, sel & ~cd, top0, cnt);
				top1 = _mm512_mask_max_epu8(top1, sel & cd, top1, cnt);
				most0 = _mm512_mask_max_epu8(most0, valid & ~cd, most0, cnt);
				most1 = _mm512_mask_max_epu8(most1, valid & cd, most1, cnt);
			}
			f.sel[k] = sel;
			f.cand[k] = cd;
			return false;
		};
#if defined(__AVX512VBMI__) && defined(__AVX512VBMI2__) && defined(__BMI2__)
		// Hard mode often allows a few guesses per block: pack runs of blocks with at most
		// 64 allowed guesses into one chunk, so each counts once.
		int A = 0, B = 0;
		if (!in.dense && n >= 8)
			for (int b = 0; b < W; b++)
			{
				A += std::popcount(in.allow[b]);
				B += in.allow[b] != 0;
			}
		if (B && A <= 40 * B)
		{
			f.packed = true;
			if ((int)f.ids.size() < W * 64) f.ids.resize((size_t)W * 64);
			const uint8_t * src[VEC_MAX];
			for (int j = 0; j < n; j++)
			{
				src[j] = in.pm + (size_t)in.s[j] * in.stride;
				sc.rows[j] = (const uint8_t *)&sc.stage[j];
			}
			for (int j = n; j < padn; j++) sc.rows[j] = sc.rows[0];
			const __m512i iota = _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
				41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
				10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
			int k = 0;
			for (int b = 0; b < W; k++)
			{
				// Chunk k takes whole blocks while their allowed guesses fit: block blk[t]'s
				// go to the lanes in put[t], picked out of its row block by idx[t].
				int blk[64], nb = 0, pos = 0;
				__m512i idx[64];
				uint64_t put[64], cd = 0;
				for (; b < W; b++)
				{
					const uint64_t a = in.allow[b];
					const int p = std::popcount(a);
					if (pos + p > 64) break;
					if (!p) continue;
					uint16_t * id = &f.ids[k * 64 + pos];
					for (uint64_t m = a; m; m &= m - 1) *id++ = (uint16_t)(b * 64 + std::countr_zero(m));
					cd |= _pext_u64(in.cand[b], a) << pos;
					put[nb] = (p == 64 ? ~0ull : (1ull << p) - 1) << pos;
					idx[nb] = _mm512_maskz_expand_epi8(put[nb], _mm512_maskz_compress_epi8(a, iota));
					blk[nb++] = b;
					pos += p;
				}
				if (!nb) break;
				for (int j = 0; j < n; j++)
				{
					__m512i row = _mm512_setzero_si512();
					for (int t = 0; t < nb; t++)
						row = _mm512_mask_permutexvar_epi8(row, put[t], idx[t], _mm512_load_si512(src[j] + 64 * (size_t)blk[t]));
					sc.stage[j] = row;
				}
				if (finish(k, df(sc.rows, 0, n), pos == 64 ? ~0ull : (1ull << pos) - 1, cd)) return true;
			}
			f.nch = k;
		}
		else
#endif
		{
			for (int j = 0; j < n; j++) sc.rows[j] = in.pm + (size_t)in.s[j] * in.stride;
			for (int j = n; j < padn; j++) sc.rows[j] = sc.rows[0];
			for (int b = 0; b < W; b++)
			{
				if (!in.allow[b]) { f.sel[b] = 0; continue; }
				if (finish(b, df(sc.rows, (size_t)b * 64, n), in.allow[b], in.cand[b])) return true;
			}
		}
		// Selected counts are at least 2, so 0 means none.
		f.top[0] = max_byte(top0) ? max_byte(top0) : -1;
		f.top[1] = max_byte(top1) ? max_byte(top1) : -1;
		f.most[0] = max_byte(most0);
		f.most[1] = max_byte(most1);
		return false;
	}
#endif
	f.top[0] = f.top[1] = -1;
	f.most[0] = f.most[1] = 0;
	for (int b = 0; b < W; b++)
	{
		const uint64_t cd = in.cand[b];
		uint64_t sel = 0;
		for (uint64_t m = in.allow[b]; m; m &= m - 1)
		{
			const int l = std::countr_zero(m);
			const uint8_t * row = in.eng.row(b * 64 + l);
			uint64_t seen[4] = { 0, 0, 0, 0 };
			for (int j = 0; j < n; j++)
			{
				const uint8_t x = row[in.s[j]];
				seen[x >> 6] |= 1ull << (x & 63);
			}
			const int d = std::popcount(seen[0]) + std::popcount(seen[1]) + std::popcount(seen[2]) + std::popcount(seen[3]);
			const int c = (int)(cd >> l & 1);
			if (c ? d == n - 1 && left >= 3 : d == n) return true;
			f.most[c] = std::max(f.most[c], d);
			if (left >= 3 && d > (c ? lo : thr))
			{
				f.cnt[b].v[l] = (uint8_t)d;
				sel |= 1ull << l;
				f.top[c] = std::max(f.top[c], d);
			}
		}
		f.sel[b] = sel;
		f.cand[b] = cd;
	}
	return false;
}

// Calls take(lb, g) for the selected guesses in the order solve tries them: by bound, then
// non-candidates first (one more group, so smaller ones), then guess order. Stops when take
// returns false. Each pass collects one count of one class and finds the next.
template <class Take>
void for_options(const Frame & f, int n, Take && take)
{
	int top[2] = { f.top[0], f.top[1] };
	while (top[0] >= 0 || top[1] >= 0)
	{
		// Bound 3n - in - count.
		const int in = top[0] < 0 ? 1 : top[1] < 0 ? 0 : 3 * n - top[0] <= 3 * n - 1 - top[1] ? 0 : 1;
		const int t = top[in], lb = 3 * n - in - t;
		int next = -1;
#ifdef __AVX512BW__
		const __m512i vt = _mm512_set1_epi8((char)t);
		__m512i nx = _mm512_setzero_si512();
#endif
		for (int k = 0; k < f.nch; k++)
		{
			const uint64_t m = f.sel[k] & (in ? f.cand[k] : ~f.cand[k]);
			if (!m) continue;
#ifdef __AVX512BW__
			const __m512i v = _mm512_load_si512(f.cnt[k].v);
			uint64_t eq = _mm512_cmpeq_epi8_mask(v, vt) & m;
			nx = _mm512_mask_max_epu8(nx, _mm512_cmplt_epu8_mask(v, vt) & m, nx, v);
#else
			uint64_t eq = 0;
			for (uint64_t r = m; r; r &= r - 1)
			{
				const int l = std::countr_zero(r), c = f.cnt[k].v[l];
				if (c == t) eq |= 1ull << l;
				else if (c < t) next = std::max(next, c);
			}
#endif
			for (; eq; eq &= eq - 1)
				if (!take(lb, f.guess(k, std::countr_zero(eq)))) return;
		}
#ifdef __AVX512BW__
		if (const int x = max_byte(nx)) next = x;
#endif
		top[in] = next;
	}
}

}

// Pool-major copy of the feedback table, rows padded to whole 64-guess blocks, and
// per-letter guess bitmasks for building a hard-mode rule's allowed set.
struct ExactSolver::Tables
{
	int G = 0, W = 0;   // guesses, 64-guess blocks
	size_t stride = 0;  // W * 64
	const uint8_t * pm = nullptr; // Engine::pool_major()
	std::vector<uint64_t> full;    // the real guesses of each block
	std::vector<uint64_t> green;   // [(pos * 26 + letter) * W + block]
	std::vector<uint64_t> atleast; // [(letter * 5 + copies - 1) * W + block]
	bool letters_ok = true;        // every word is a-z

	const uint8_t * prow(int p) const { return pm + (size_t)p * stride; }

	void allowed(const HardRule & r, const std::vector<std::string> & words, uint64_t * out) const
	{
		if (!letters_ok)
		{
			std::fill(out, out + W, 0);
			for (int g = 0; g < G; g++)
				if (r.allows(words[g].data())) out[g >> 6] |= 1ull << (g & 63);
			return;
		}
		std::copy(full.begin(), full.end(), out);
		auto apply = [&](const uint64_t * m) { for (int b = 0; b < W; b++) out[b] &= m[b]; };
		for (int i = 0; i < 5; i++)
			if (r.green[i]) apply(&green[(size_t)(i * 26 + (r.green[i] - 'a')) * W]);
		for (int k = 0; k < 5 && r.letter[k]; k++)
			if (r.count[k]) apply(&atleast[(size_t)((r.letter[k] - 'a') * 5 + r.count[k] - 1) * W]);
	}
};

struct ExactSolver::Memo
{
	static constexpr int SHARDS = 64;
	static constexpr size_t MAX_PER_SHARD = 1 << 16;
	struct Entry { int value; bool exact; };
	struct Shard { std::mutex mu; std::unordered_map<std::string, Entry> map; } shard[SHARDS];

	Shard & of(const std::string & k) { return shard[std::hash<std::string>()(k) % SHARDS]; }
	bool get(const std::string & k, Entry & e)
	{
		Shard & s = of(k);
		std::lock_guard<std::mutex> lock(s.mu);
		auto it = s.map.find(k);
		if (it == s.map.end()) return false;
		e = it->second;
		return true;
	}
	void put(const std::string & k, Entry e)
	{
		Shard & s = of(k);
		std::lock_guard<std::mutex> lock(s.mu);
		if (s.map.size() >= MAX_PER_SHARD) s.map.clear();
		auto [it, fresh] = s.map.emplace(k, e);
		if (!fresh && !it->second.exact && (e.exact || e.value > it->second.value)) it->second = e;
	}
};

ExactSolver::ExactSolver(const Engine & eng, bool hard, int threads)
	: eng_(eng), hard_(hard), threads_(threads > 0 ? threads : (int)std::max(1u, std::thread::hardware_concurrency())),
	  memo_(new Memo), tab_(new Tables)
{
	Tables & t = *tab_;
	const int G = eng.num_guesses(), P = eng.pool_size();
	t.G = G;
	t.W = (G + 63) / 64;
	t.stride = (size_t)t.W * 64;
	// Padding guesses read as 0 and are never allowed.
	size_t stride;
	t.pm = eng.pool_major(stride);
	if (stride != t.stride) throw std::logic_error("pool-major table stride mismatch");

	t.full.assign(t.W, ~0ull);
	if (G % 64) t.full[t.W - 1] = (1ull << (G % 64)) - 1;
	t.green.assign((size_t)5 * 26 * t.W, 0);
	t.atleast.assign((size_t)26 * 5 * t.W, 0);
	const auto & words = eng.words();
	for (int g = 0; g < G && t.letters_ok; g++)
	{
		const char * w = words[g].data();
		const uint64_t bit = 1ull << (g & 63);
		for (int i = 0; i < 5; i++)
		{
			if (w[i] < 'a' || w[i] > 'z') { t.letters_ok = false; break; }
			t.green[(size_t)(i * 26 + (w[i] - 'a')) * t.W + (g >> 6)] |= bit;
			int copies = 0;
			for (int j = 0; j < 5; j++) copies += w[j] == w[i];
			for (int m = 1; m <= copies; m++) t.atleast[(size_t)((w[i] - 'a') * 5 + m - 1) * t.W + (g >> 6)] |= bit;
		}
	}
}

ExactSolver::~ExactSolver() = default;

int ExactSolver::solve(const uint16_t * s, int n, int left, const HardRule & rule, int beta)
{
	if (left <= 0) return INF;
	if (n == 1) return 1;
	if (left == 1) return INF;
	if (n == 2) return 3;
	// Every word spends a guess and all but one at least two: no strategy beats 2n-1.
	if (beta <= 2 * n - 1) return 2 * n - 1;
	if (cancel_ && cancel_->load(std::memory_order_relaxed)) throw Cancelled();

	const std::string key = make_key(s, n, left, hard_ ? &rule : nullptr);
	Memo::Entry e;
	if (memo_->get(key, e) && (e.exact || e.value >= beta)) return e.value;

	const auto & pool = eng_.pool();
	const auto & words = eng_.words();

	// Guessing a candidate that tells all the others apart is optimal: 2n-1.
	if (perfect_split(eng_, s, n))
	{
		memo_->put(key, { 2 * n - 1, true });
		return 2 * n - 1;
	}

	// Otherwise every guess has bound >= 2n: a candidate splits the rest into at most
	// n-2 groups, any other guess into at most n.
	if (beta <= 2 * n) return 2 * n;

	// Scan every allowed guess; see scan().
	const Tables & t = *tab_;
	Scratch & sc = tl_scratch;
	const int W = t.W;
	const uint64_t * allow = t.full.data();
	if (hard_)
	{
		sc.allow.resize(W);
		t.allowed(rule, words, sc.allow.data());
		allow = sc.allow.data();
	}
	sc.cand.assign(W, 0);
	for (int j = 0; j < n; j++) sc.cand[pool[s[j]] >> 6] |= 1ull << (pool[s[j]] & 63);
	// Guesses with more distinct feedbacks than this may have bound < beta (candidates
	// one less), and each must have at least two.
	const int lo = left == 2 ? n - 1 : std::max(1, 3 * n - 1 - beta);
	FrameRef fr(W);
	Frame & f = fr.f;
	if (scan({ eng_, t.pm, t.stride, W, s, n, left, beta, lo, allow, sc.cand.data(), allow == t.full.data() }, f))
	{
		memo_->put(key, { 2 * n, true });
		return 2 * n;
	}
	if (left == 2)
	{
		memo_->put(key, { INF, true });
		return INF;
	}
	int km = f.most[1], kn = f.most[0]; // most distinct feedbacks of a candidate guess / of any other
	// The caps bound may refute the node, or show it can't be done in time.
	int caps = -1;
	if (n <= CapBound::T && left <= CapBound::H)
	{
		km = std::min(km, CapBound::T);
		kn = std::min(kn, CapBound::T);
		const int b = cap_bound().root[CapBound::at(left, km, kn, n)];
		if (b >= INF)
		{
			memo_->put(key, { INF, true });
			return INF;
		}
		if (b >= beta)
		{
			memo_->put(key, { b, false });
			return b;
		}
		caps = km << 8 | kn;
	}

	int best = beta;
	if (n < SHARE_MIN || threads_ == 1)
	{
		for_options(f, n, [&](int lb, int g) {
			if (lb >= best) return false;
			best = std::min(best, try_option(s, n, left, rule, { lb, g }, best, caps));
			return true;
		});
	}
	else
	{
		// Big node: let idle threads take options too, all pruning against one best.
		std::vector<Opt> opts;
		for_options(f, n, [&](int lb, int g) { opts.push_back({ lb, g }); return true; });
		Shared sh{ s, n, left, &rule, &opts };
		sh.caps = caps;
		sh.best = beta;
		{
			std::lock_guard<std::mutex> lock(shared_mu_);
			shared_.push_back(&sh);
		}
		struct Leave
		{
			ExactSolver & x;
			Shared & sh;
			~Leave()
			{
				std::unique_lock<std::mutex> lock(x.shared_mu_);
				x.shared_.erase(std::find(x.shared_.begin(), x.shared_.end(), &sh));
				x.shared_cv_.wait(lock, [&] { return sh.helpers == 0; });
			}
		} leave{ *this, sh };
		work_on(sh);
		best = sh.best;
	}
	memo_->put(key, { best, best < beta });
	return best;
}

int ExactSolver::try_option(const uint16_t * s, int n, int left, const HardRule & rule, const Opt & o, int best, int caps)
{
	Hist & h = tl_hist;
	const uint8_t * row = eng_.row(o.g);
	h.reset();
	for (int j = 0; j < n; j++) h.add(row[s[j]]);
	// Groups of one or two words take exactly 2s-1 (left >= 3 here), so they are left
	// out. The others go smallest first, (size, pattern) packed into one integer: small
	// groups are cheap to solve and refute most options long before a big one would.
	uint32_t groups[NPAT];
	int ng = 0;
	for (int k = 0; k < h.nused; k++)
	{
		const uint8_t f = h.used[k];
		if (f != ALL_GREEN && h.cnt[f] >= 3) groups[ng++] = (uint32_t)h.cnt[f] << 8 | f;
	}
	std::sort(groups, groups + ng);
	// total: what is settled exactly; rest: lower bounds of the groups still open.
	// o.lb charges every group 2s-1.
	int total = o.lb, rest = 0;
	int off[NPAT], start[NPAT], at = 0;
	for (int i = 0; i < ng; i++)
	{
		const int size = groups[i] >> 8;
		off[i] = start[groups[i] & 255] = at;
		at += size;
		total -= 2 * size - 1;
		rest += 2 * size - 1;
	}
	// Children of a shared node read their set from here, so it must not move until we return.
	constexpr int STACK_PART = 1024;
	uint16_t stack_part[STACK_PART];
	std::unique_ptr<uint16_t[]> heap_part;
	uint16_t * part = stack_part;
	if (at > STACK_PART) { heap_part.reset(new uint16_t[at]); part = heap_part.get(); }
	for (int j = 0; j < n; j++)
	{
		const uint8_t f = row[s[j]];
		if (f != ALL_GREEN && h.cnt[f] >= 3) part[start[f]++] = s[j];
	}

	// A group some candidate splits apart takes exactly 2s-1, as solve would find;
	// any other at least 2s, and at least what this node's caps allow (see CapBound).
	// That alone refutes many options.
	int open[NPAT], open_lb[NPAT], nopen = 0;
	const CapBound & cb = cap_bound();
	for (int i = 0; i < ng; i++)
	{
		const int size = groups[i] >> 8;
		if (perfect_split(eng_, part + off[i], size)) { total += 2 * size - 1; rest -= 2 * size - 1; }
		else
		{
			int lb = 2 * size;
			if (caps >= 0 && size <= CapBound::T) lb = std::max(lb, cb.sub[CapBound::at(left - 1, caps >> 8, caps & 255, size)]);
			if (lb >= INF) return best;
			open[nopen] = i;
			open_lb[nopen++] = lb;
			rest += lb - (2 * size - 1);
			if (total + rest >= best) return best;
		}
	}
	for (int k = 0; k < nopen; k++)
	{
		const int i = open[k];
		const int size = groups[i] >> 8;
		rest -= open_lb[k];
		HardRule sub = rule;
		if (hard_) sub.add(eng_.words()[o.g].data(), groups[i] & 255);
		total += solve(part + off[i], size, left - 1, sub, best - total - rest);
		if (total + rest >= best) return best;
	}
	return total;
}

void ExactSolver::work_on(Shared & sh)
{
	const size_t count = sh.opts->size();
	for (size_t i; (i = sh.next++) < count;)
	{
		const Opt & o = (*sh.opts)[i];
		const int best = sh.best.load();
		// Options are sorted by bound, so none after this one can win either.
		if (o.lb >= best) { sh.next = count; break; }
		const int t = try_option(sh.s, sh.n, sh.left, *sh.rule, o, best, sh.caps);
		for (int cur = sh.best.load(); t < cur && !sh.best.compare_exchange_weak(cur, t);) {}
	}
}

bool ExactSolver::help()
{
	Shared * sh = nullptr;
	{
		std::lock_guard<std::mutex> lock(shared_mu_);
		// The biggest node has the longest options left.
		for (Shared * x : shared_)
			if (x->next < x->opts->size() && (!sh || x->n > sh->n)) sh = x;
		if (!sh) return false;
		sh->helpers++;
	}
	struct Done
	{
		ExactSolver & x;
		Shared * sh;
		~Done()
		{
			std::lock_guard<std::mutex> lock(x.shared_mu_);
			if (--sh->helpers == 0) x.shared_cv_.notify_all();
		}
	} done{ *this, sh };
	work_on(*sh);
	return true;
}

bool ExactSolver::totals(const std::vector<uint16_t> & cands, const std::vector<int> & g1s, int left,
	const HardRule & rule, std::vector<int> & out,
	const std::function<void(double)> & progress, const std::atomic<bool> * cancel,
	const std::vector<int> * beta, const std::vector<const int *> * group_k)
{
	cancel_ = cancel;
	const int n = (int)cands.size();

	// Every group starts at a lower bound, 2s-1 or the lookahead's 3s-k; sum[i] is guess
	// i's total with each group at its bound, or at its value once solved. Values only
	// rise, so once sum[i] reaches beta[i] the guess is refuted and its other groups are skipped.
	struct Task { int idx, lb; std::vector<uint16_t> set; HardRule rule; };
	std::vector<Task> tasks;
	std::vector<std::atomic<int>> sum(g1s.size());
	for (size_t i = 0; i < g1s.size(); i++)
	{
		const int g = g1s[i];
		const uint8_t * row = eng_.row(g);
		const int * gk = group_k ? (*group_k)[i] : nullptr;
		std::vector<std::vector<uint16_t>> parts(NPAT);
		for (uint16_t c : cands) parts[row[c]].push_back(c);
		int total = n;
		for (int p = 0; p < ALL_GREEN; p++)
		{
			const int s = (int)parts[p].size();
			if (s == 0) continue;
			int lb = 2 * s - 1;
			if (gk && s >= 3 && gk[p] > 0) lb = std::max(lb, 3 * s - gk[p]);
			total += lb;
			HardRule sub = rule;
			if (hard_) sub.add(eng_.words()[g].data(), p);
			tasks.push_back({ (int)i, lb, std::move(parts[p]), sub });
		}
		sum[i] = std::min(total, INF);
	}
	if (beta)
		// Small groups first: they are cheap, and usually lift the total past beta (or show
		// it impossible) before a big group has to be solved at all.
		std::stable_sort(tasks.begin(), tasks.end(), [](const Task & a, const Task & b) { return a.set.size() < b.set.size(); });
	else
		// Big groups first so the threads finish together.
		std::sort(tasks.begin(), tasks.end(), [](const Task & a, const Task & b) { return a.set.size() > b.set.size(); });

	std::atomic<size_t> next{ 0 }, done{ 0 };
	std::atomic<bool> cancelled{ false };
	std::vector<char> ran(tasks.size(), 0);
	std::mutex progress_mu;
	auto work = [&] {
		try
		{
			for (;;)
			{
				const size_t t = next++;
				if (t < tasks.size())
				{
					const Task & k = tasks[t];
					std::atomic<int> & acc = sum[k.idx];
					// The value at which this group alone would lift the total to beta.
					int b = INF;
					if (beta)
					{
						const int cur = acc.load();
						b = cur >= (*beta)[k.idx] ? -1 : (*beta)[k.idx] - (cur - k.lb);
					}
					if (b >= 0)
					{
						const int v = solve(k.set.data(), (int)k.set.size(), left - 1, k.rule, b);
						for (int cur = acc.load(); !acc.compare_exchange_weak(cur, std::min(INF, cur + v - k.lb));) {}
					}
					ran[t] = 1;
					const size_t d = ++done;
					if (progress)
					{
						std::lock_guard<std::mutex> lock(progress_mu);
						progress((double)d / tasks.size());
					}
					continue;
				}
				// Out of tasks: help the ones still running until they are all done.
				if (cancelled || done == tasks.size()) break;
				if (!help()) std::this_thread::sleep_for(std::chrono::microseconds(200));
			}
		}
		catch (const Cancelled &) { cancelled = true; }
	};
	std::vector<std::thread> pool;
	for (int t = 0; t < threads_; t++) pool.emplace_back(work);
	for (auto & t : pool) t.join();
	cancel_ = nullptr;

	// Each group either came in below its target (exact) or lifted the sum to beta, so the
	// sum is exact below beta and a bound from there on. After a cancel, guesses with every
	// group done, or already refuted, still get their value.
	out.assign(g1s.size(), 0);
	for (size_t i = 0; i < g1s.size(); i++) out[i] = sum[i].load();
	for (size_t t = 0; t < tasks.size(); t++)
	{
		const int i = tasks[t].idx;
		if (!ran[t] && !(beta && out[i] >= (*beta)[i])) out[i] = -1;
	}
	return !cancelled;
}
