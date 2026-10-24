#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#define CUDA_CHECK(x) do { \
	cudaError_t err_ = (x); \
	if (err_ != cudaSuccess) \
		throw std::runtime_error(std::string(#x ": ") + cudaGetErrorString(err_)); \
} while (0)

constexpr int BD = 256;             // threads per block in every kernel
constexpr double LAUNCH_MS = 25;    // target length of one lookahead launch

// Row stride of the pool-major table: a multiple of 128 bytes, padded with zeros.
static int padded(int G) { return (G + 127) & ~127; }

// Feedback for every (answer, guess) pair, in both layouts: pool-major for the
// lookahead kernel (a warp reads 32 neighbouring guesses for one answer) and
// guess-major for the stats kernel and the host.
__global__ void table_kernel(const char * __restrict__ words, const int * __restrict__ pool,
	int G, int GP, int P, uint8_t * pt, uint8_t * tt)
{
	const int g = blockIdx.x * blockDim.x + threadIdx.x;
	const int p = blockIdx.y;
	if (g >= G) return;
	const uint8_t f = feedback(words + 5 * g, words + 5 * pool[p]);
	pt[(size_t)p * GP + g] = f;
	tt[(size_t)g * P + p] = f;
}

template <typename T, typename Op>
__device__ T block_reduce(T v, Op op, T * scratch)
{
	for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffff, v, o));
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	__syncthreads();
	if (lane == 0) scratch[warp] = v;
	__syncthreads();
	v = scratch[0];
	for (int w = 1; w < BD / 32; w++) v = op(v, scratch[w]);
	return v;
}

struct Max { __device__ int operator()(int a, int b) const { return a > b ? a : b; } };
struct Sum { template <typename T> __device__ T operator()(T a, T b) const { return a + b; } };

__global__ void __launch_bounds__(BD) stats_kernel(const uint8_t * __restrict__ tt, int P,
	const uint16_t * __restrict__ cands, int n, const int * __restrict__ guesses, GuessStats * out)
{
	__shared__ int hist[NPAT];
	__shared__ int iscratch[BD / 32];
	__shared__ float fscratch[BD / 32];
	const int tid = threadIdx.x;
	const uint8_t * row = tt + (size_t)guesses[blockIdx.x] * P;

	for (int i = tid; i < NPAT; i += BD) hist[i] = 0;
	__syncthreads();
	for (int i = tid; i < n; i += BD) atomicAdd(&hist[row[cands[i]]], 1);
	__syncthreads();

	int buckets = 0, largest = 0, sum_sq = 0;
	float clogc = 0;
	for (int i = tid; i < NPAT; i += BD)
	{
		const int c = hist[i];
		buckets += c > 0;
		largest = max(largest, c);
		sum_sq += c * c;
		if (c > 1) clogc += c * log2f((float)c);
	}
	buckets = block_reduce(buckets, Sum(), iscratch);
	largest = block_reduce(largest, Max(), iscratch);
	sum_sq = block_reduce(sum_sq, Sum(), iscratch);
	clogc = block_reduce(clogc, Sum(), fscratch);
	if (tid == 0)
		out[blockIdx.x] = { buckets, largest, sum_sq, log2f((float)n) - clogc / n, hist[ALL_GREEN] > 0 };
}

// Second guesses per thread in lookahead_kernel: adjacent bytes of a table row, read as one word.
constexpr int LV = 4;
// Members whose table words are in flight at once per thread.
constexpr int DEPTH = 8;

// Blocks (x, y) take first guess x and the y-th slice of second guesses. The candidates
// are grouped by their feedback to the first guess; for every group of 3+ each thread
// tries LV adjacent second guesses at once, counting the distinct feedbacks of each in a
// 243-bit set held in its own columns of shared memory. Per-group maxima over the slices
// meet in best_d / best_k. Rows of pt are GP bytes (a multiple of 128) and slices are
// multiples of 32 quads, so a warp reads one aligned 128-byte line per member.

// Set the bits for one member's four patterns. The four sets are disjoint, so all four
// words are read before any is written.
// mul is LV * BD * 4 / 32 = 128, passed in so the compiler keeps (q & 0xe0) * mul + base
// as LOP3 + IMAD (FMA pipe) instead of SHF + LOP3 + IADD (all on the ALU pipe).
__device__ __forceinline__ void set4(uint32_t * mybits, uint32_t v, uint32_t mul)
{
	// Byte k of v is the pattern q of second guess k; the word holding bit q is q >> 5 at a
	// stride of LV * BD words, i.e. (q & 0xe0) * 128 bytes. Shifts by y use its low 5 bits.
	const uint32_t y1 = v >> 8, y2 = v >> 16, y3 = v >> 24;
	char * base = (char *)mybits;
	uint32_t * a0 = (uint32_t *)(base + (v & 0xe0) * mul);
	uint32_t * a1 = (uint32_t *)(base + (y1 & 0xe0) * mul) + BD;
	uint32_t * a2 = (uint32_t *)(base + (y2 & 0xe0) * mul) + 2 * BD;
	uint32_t * a3 = (uint32_t *)(base + (y3 & 0xe0) * mul) + 3 * BD;
#ifdef USE_ATOMS
	// Fire-and-forget shared atomics: no load-to-store chain between members.
	atomicOr(a0, 1u << (v & 31));
	atomicOr(a1, 1u << (y1 & 31));
	atomicOr(a2, 1u << (y2 & 31));
	atomicOr(a3, 1u << (y3 & 31));
#else
	const uint32_t w0 = *a0, w1 = *a1, w2 = *a2, w3 = *a3;
	*a0 = w0 | 1u << (v & 31);
	*a1 = w1 | 1u << (y1 & 31);
	*a2 = w2 | 1u << (y2 & 31);
	*a3 = w3 | 1u << (y3 & 31);
#endif
}

template <bool HARD>
__global__ void __launch_bounds__(BD) lookahead_kernel(const uint8_t * __restrict__ pt, int G, int GP, int per,
	const char * __restrict__ words, const uint16_t * __restrict__ cands, int n,
	const int * __restrict__ g1s, const int * __restrict__ g2cand, HardRule base,
	int * group_size, int * best_d, int * best_k)
{
	extern __shared__ uint32_t smem[];
	uint32_t * bits = smem;                              // [8][LV][BD]
	int * counts = (int *)(bits + 8 * LV * BD);          // [256]
	int * starts = counts + 256;                         // [256]
	int * scratch = starts + 256;                        // [2][BD/32]
	uint16_t * members = (uint16_t *)(scratch + 2 * BD / 32); // [n] pool indices grouped by feedback
	uint8_t * pat1 = (uint8_t *)(members + n);           // [n] feedback of each candidate
	__shared__ int perfect;

	const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	const int g1 = g1s[blockIdx.x];
	const int quads = GP / LV;
	const int q_lo = blockIdx.y * per, q_hi = min(quads, q_lo + per);
	group_size += blockIdx.x * NPAT;
	best_d += blockIdx.x * NPAT;
	best_k += blockIdx.x * NPAT;

	for (int i = tid; i < 256; i += BD) counts[i] = 0;
	__syncthreads();
	for (int i = tid; i < n; i += BD)
	{
		const uint8_t f = pt[(uint32_t)cands[i] * GP + g1];
		pat1[i] = f;
		atomicAdd(&counts[f], 1);
	}
	__syncthreads();
	if (tid == 0)
	{
		int acc = 0;
		for (int i = 0; i < 256; i++) { starts[i] = acc; acc += counts[i]; }
	}
	__syncthreads();
	// Use starts as cursors while scattering, then restore them from counts.
	for (int i = tid; i < n; i += BD)
		members[atomicAdd(&starts[pat1[i]], 1)] = cands[i];
	__syncthreads();
	for (int i = tid; i < NPAT; i += BD)
	{
		starts[i] -= counts[i];
		if (blockIdx.y == 0) group_size[i] = counts[i];
	}
	__syncthreads();

	uint32_t * mybits = bits + tid;
	for (int p = 0; p < ALL_GREEN; p++)
	{
		const int s = counts[p];
		// Groups of 1 or 2 are split perfectly by guessing one of their words.
		if (s < 3) continue;
		// Another slice may already have split this group perfectly. Thread 0 decides for
		// the whole block so every thread reaches the same barriers. This barrier also
		// keeps the previous group's scratch until thread 0 has read it.
		if (tid == 0) perfect = *(volatile int *)&best_k[p] == s + 1;
		__syncthreads();

		HardRule rule = base;
		if (HARD) rule.add(words + 5 * g1, p);
		const uint16_t * mem = members + starts[p];
		const uint32_t mul = LV * BD * 4 / 32 + (GP >> 30); // GP < 2^30: just hides the constant
		int bd = 0, bk = 0;
		for (int qd = q_lo + tid; qd < q_hi; qd += BD)
		{
			if (*(volatile int *)&perfect) break;
			const int g2 = qd * LV;
			unsigned ok = 0; // bit k: g2 + k is a word, and one the rules allow
#pragma unroll
			for (int k = 0; k < LV; k++)
				if (g2 + k < G && (!HARD || rule.allows(words + 5 * (g2 + k)))) ok |= 1u << k;
			if (!ok) continue;
#pragma unroll
			for (int w = 0; w < 8 * LV; w++) mybits[w * BD] = 0;
			// Fetch the next DEPTH members' patterns before setting the bits of these: the bit
			// stores might alias the member list as far as the compiler knows, so it will not
			// hoist loads past them itself. All tests on j and s are block-uniform.
			const uint32_t * col = (const uint32_t *)pt + qd;
			const uint32_t stride = GP / 4;
			uint32_t v[DEPTH], nx[DEPTH];
#pragma unroll
			for (int u = 0; u < DEPTH; u++) v[u] = u < s ? col[mem[u] * stride] : 0;
			int j = 0;
			for (; j + 2 * DEPTH <= s; j += DEPTH) // the next batch is full
			{
#pragma unroll
				for (int u = 0; u < DEPTH; u++) nx[u] = col[mem[j + DEPTH + u] * stride];
#pragma unroll
				for (int u = 0; u < DEPTH; u++) set4(mybits, v[u], mul);
#pragma unroll
				for (int u = 0; u < DEPTH; u++) v[u] = nx[u];
			}
			if (j + DEPTH < s) // a partial batch follows
			{
#pragma unroll
				for (int u = 0; u < DEPTH; u++) nx[u] = j + DEPTH + u < s ? col[mem[j + DEPTH + u] * stride] : 0;
#pragma unroll
				for (int u = 0; u < DEPTH; u++) set4(mybits, v[u], mul);
#pragma unroll
				for (int u = 0; u < DEPTH; u++) v[u] = nx[u];
				j += DEPTH;
			}
#pragma unroll
			for (int u = 0; u < DEPTH; u++)
				if (j + u < s) set4(mybits, v[u], mul);
			const int4 c4 = *(const int4 *)(g2cand + g2);
			const int cs[LV] = { c4.x, c4.y, c4.z, c4.w };
#pragma unroll
			for (int k = 0; k < LV; k++)
			{
				int d = 0;
#pragma unroll
				for (int w = 0; w < 8; w++) d += __popc(mybits[(w * LV + k) * BD]);
				if (!(ok >> k & 1)) continue;
				const int c = cs[k];
				const int kk = d + (c >= 0 && pat1[c] == p);
				bd = max(bd, d);
				bk = max(bk, kk);
				if (kk == s + 1) perfect = 1;
			}
		}
		for (int o = 16; o > 0; o >>= 1)
		{
			bd = max(bd, __shfl_xor_sync(0xffffffff, bd, o));
			bk = max(bk, __shfl_xor_sync(0xffffffff, bk, o));
		}
		if (lane == 0) { scratch[warp] = bd; scratch[BD / 32 + warp] = bk; }
		__syncthreads();
		if (tid == 0)
		{
			for (int w = 1; w < BD / 32; w++) { bd = max(bd, scratch[w]); bk = max(bk, scratch[BD / 32 + w]); }
			atomicMax(&best_d[p], bd);
			atomicMax(&best_k[p], bk);
		}
	}
}

// One thread per first guess: add up the groups.
__global__ void lookahead_sum(int count, const int * __restrict__ group_size, const int * __restrict__ best_d,
	const int * __restrict__ best_k, Lookahead * out)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) return;
	const int * size = group_size + i * NPAT;
	int total = 0, by3 = 0;
	if (size[ALL_GREEN]) total = by3 = 1;
	for (int p = 0; p < ALL_GREEN; p++)
	{
		const int s = size[p];
		if (s == 0) continue;
		const int d = s < 3 ? s : best_d[i * NPAT + p];
		const int k = s < 3 ? s + 1 : best_k[i * NPAT + p];
		// Each word spends guesses 1 and 2; a group of s left after that needs at least
		// 2s-1 more, and the second guess itself, if it was the answer, needs none.
		total += 4 * s - k;
		by3 += d;
	}
	out[i] = { total, by3 };
}

template <typename T>
struct DevBuf
{
	T * p = nullptr;
	size_t cap = 0;
	~DevBuf() { cudaFree(p); }
	void reserve(size_t n)
	{
		if (n <= cap) return;
		cudaFree(p);
		p = nullptr;
		CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
		cap = n;
	}
	void upload(const std::vector<T> & v)
	{
		reserve(v.size());
		CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
	}
};

struct Engine::Device
{
	std::mutex mu;
	DevBuf<char> words;
	DevBuf<uint8_t> pt, tt;
	DevBuf<uint16_t> cands;
	DevBuf<int> guesses, g2cand;
	DevBuf<GuessStats> stats;
	DevBuf<Lookahead> look;
	DevBuf<int> group_size, best_d, best_k;
	cudaEvent_t start, stop;
};

Engine::Engine(std::vector<std::string> words, std::vector<int> pool, int device)
	: dev_(new Device), words_(std::move(words)), pool_(std::move(pool))
{
	const int G = num_guesses(), P = pool_size();
	if (G == 0 || P == 0 || G >= 65536 || P >= 65536)
		throw std::runtime_error("word lists must be non-empty and under 65536 words");

	CUDA_CHECK(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
	device_name_ = prop.name;
	CUDA_CHECK(cudaEventCreate(&dev_->start));
	CUDA_CHECK(cudaEventCreate(&dev_->stop));

	std::vector<char> flat(5 * G);
	for (int i = 0; i < G; i++) std::copy_n(words_[i].data(), 5, &flat[5 * i]);
	dev_->words.upload(flat);
	DevBuf<int> dpool;
	dpool.upload(pool_);
	const int GP = padded(G);
	dev_->pt.reserve((size_t)GP * P);
	dev_->tt.reserve((size_t)G * P);
	CUDA_CHECK(cudaMemset(dev_->pt.p, 0, (size_t)GP * P));
	table_kernel<<<dim3((G + BD - 1) / BD, P), BD>>>(dev_->words.p, dpool.p, G, GP, P, dev_->pt.p, dev_->tt.p);
	CUDA_CHECK(cudaGetLastError());
	table_.resize((size_t)G * P);
	CUDA_CHECK(cudaMemcpy(table_.data(), dev_->tt.p, table_.size(), cudaMemcpyDeviceToHost));

	const size_t max_smem = (8 * LV * BD + 512 + 2 * BD / 32) * 4 + 3 * (size_t)P;
	CUDA_CHECK(cudaFuncSetAttribute(lookahead_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_smem));
	CUDA_CHECK(cudaFuncSetAttribute(lookahead_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_smem));
}

Engine::~Engine()
{
	cudaEventDestroy(dev_->start);
	cudaEventDestroy(dev_->stop);
}

const uint8_t * Engine::pool_major(size_t & stride) const
{
	std::call_once(pm_once_, [this] {
		const size_t G = num_guesses(), P = pool_size(), st = (G + 63) / 64 * 64;
		uint8_t * p = (uint8_t *)std::aligned_alloc(64, P * st);
		if (!p) throw std::bad_alloc();
		pm_.reset(p);
		for (size_t i = 0; i < P; i++) std::fill(p + i * st + G, p + (i + 1) * st, 0);
		std::lock_guard<std::mutex> lock(dev_->mu);
		CUDA_CHECK(cudaMemcpy2D(p, st, dev_->pt.p, padded((int)G), G, P, cudaMemcpyDeviceToHost));
		pm_stride_ = st;
	});
	stride = pm_stride_;
	return pm_.get();
}

void Engine::stats(const std::vector<uint16_t> & cands, const std::vector<int> & guesses,
	std::vector<GuessStats> & out)
{
	out.resize(guesses.size());
	if (guesses.empty() || cands.empty()) return;
	std::lock_guard<std::mutex> lock(dev_->mu);
	dev_->cands.upload(cands);
	dev_->guesses.upload(guesses);
	dev_->stats.reserve(guesses.size());
	stats_kernel<<<guesses.size(), BD>>>(dev_->tt.p, pool_size(), dev_->cands.p, (int)cands.size(),
		dev_->guesses.p, dev_->stats.p);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(out.data(), dev_->stats.p, out.size() * sizeof(GuessStats), cudaMemcpyDeviceToHost));
}

bool Engine::lookahead(const std::vector<uint16_t> & cands, const std::vector<int> & g1s,
	const HardRule * hard, std::vector<Lookahead> & out,
	const std::function<void(double)> & progress, const std::atomic<bool> * cancel)
{
	out.resize(g1s.size());
	if (g1s.empty() || cands.empty()) return true;
	const int G = num_guesses(), n = (int)cands.size();

	const int GP = padded(G);
	std::vector<int> g2cand(GP, -1);
	for (int i = 0; i < n; i++) g2cand[pool_[cands[i]]] = i;

	std::lock_guard<std::mutex> lock(dev_->mu);
	Device & d = *dev_;
	d.cands.upload(cands);
	d.guesses.upload(g1s);
	d.g2cand.upload(g2cand);
	d.look.reserve(g1s.size());
	const size_t smem = (8 * LV * BD + 512 + 2 * BD / 32) * 4 + 3 * (size_t)n;
	const HardRule base = hard ? *hard : HardRule();

	// Split each first guess over enough blocks that one block takes about a millisecond,
	// then size batches so a launch takes about LAUNCH_MS.
	// Slices are whole warps of quads (32 quads = one 128-byte line per member).
	const int quads = GP / LV;
	const int want = (int)std::clamp<size_t>((size_t)n * G / (LV * 1000000), 1, 64);
	const int per = ((quads + want - 1) / want + 31) & ~31;
	const int slices = (quads + per - 1) / per;
	size_t batch = std::max(1, 4096 / slices);
	for (size_t at = 0; at < g1s.size();)
	{
		if (cancel && cancel->load()) return false;
		const size_t cnt = std::min(batch, g1s.size() - at);
		d.group_size.reserve(cnt * NPAT);
		d.best_d.reserve(cnt * NPAT);
		d.best_k.reserve(cnt * NPAT);
		CUDA_CHECK(cudaMemsetAsync(d.best_d.p, 0, cnt * NPAT * sizeof(int)));
		CUDA_CHECK(cudaMemsetAsync(d.best_k.p, 0, cnt * NPAT * sizeof(int)));
		CUDA_CHECK(cudaEventRecord(d.start));
		const dim3 grid((unsigned)cnt, slices);
		if (hard)
			lookahead_kernel<true><<<grid, BD, smem>>>(d.pt.p, G, GP, per, d.words.p, d.cands.p, n,
				d.guesses.p + at, d.g2cand.p, base, d.group_size.p, d.best_d.p, d.best_k.p);
		else
			lookahead_kernel<false><<<grid, BD, smem>>>(d.pt.p, G, GP, per, d.words.p, d.cands.p, n,
				d.guesses.p + at, d.g2cand.p, base, d.group_size.p, d.best_d.p, d.best_k.p);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaEventRecord(d.stop));
		lookahead_sum<<<(cnt + BD - 1) / BD, BD>>>((int)cnt, d.group_size.p, d.best_d.p, d.best_k.p, d.look.p + at);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaEventSynchronize(d.stop));
		float ms = 0;
		CUDA_CHECK(cudaEventElapsedTime(&ms, d.start, d.stop));
		at += cnt;
		if (cnt == batch)
			batch = std::clamp<size_t>((size_t)(batch * LAUNCH_MS / std::max(ms, 0.01f)), 64, batch * 2);
		if (progress) progress((double)at / g1s.size());
	}
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(out.data(), d.look.p, out.size() * sizeof(Lookahead), cudaMemcpyDeviceToHost));
	return true;
}
