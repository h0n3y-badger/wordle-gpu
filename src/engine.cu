#include "engine.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>

#define CUDA_CHECK(x) do { \
	cudaError_t err_ = (x); \
	if (err_ != cudaSuccess) \
		throw std::runtime_error(std::string(#x ": ") + cudaGetErrorString(err_)); \
} while (0)

constexpr int BD = 256;             // threads per block in every kernel
constexpr double LAUNCH_MS = 25;    // target length of one lookahead launch

// Feedback for every (answer, guess) pair, in both layouts: pool-major for the
// lookahead kernel (a warp reads 32 neighbouring guesses for one answer) and
// guess-major for the stats kernel and the host.
__global__ void table_kernel(const char * __restrict__ words, const int * __restrict__ pool,
	int G, int P, uint8_t * pt, uint8_t * tt)
{
	const int g = blockIdx.x * blockDim.x + threadIdx.x;
	const int p = blockIdx.y;
	if (g >= G) return;
	const uint8_t f = feedback(words + 5 * g, words + 5 * pool[p]);
	pt[(size_t)p * G + g] = f;
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

// One block per first guess. The candidates are grouped by their feedback; for every
// group of 3+ each thread tries second guesses, counting the distinct feedbacks in a
// 243-bit set held in its own column of shared memory.
template <bool HARD>
__global__ void __launch_bounds__(BD) lookahead_kernel(const uint8_t * __restrict__ pt, int G,
	const char * __restrict__ words, const uint16_t * __restrict__ cands, int n,
	const int * __restrict__ g1s, const int * __restrict__ g2cand, HardRule base, Lookahead * out)
{
	extern __shared__ uint32_t smem[];
	uint32_t * bits = smem;                              // [8][BD]
	int * counts = (int *)(bits + 8 * BD);               // [256]
	int * starts = counts + 256;                         // [256]
	int * scratch = starts + 256;                        // [BD/32]
	uint16_t * members = (uint16_t *)(scratch + BD / 32); // [n] pool indices grouped by feedback
	uint8_t * pat1 = (uint8_t *)(members + n);           // [n] feedback of each candidate
	__shared__ int perfect;

	const int tid = threadIdx.x;
	const int g1 = g1s[blockIdx.x];

	for (int i = tid; i < 256; i += BD) counts[i] = 0;
	if (tid == 0) perfect = 0;
	__syncthreads();
	for (int i = tid; i < n; i += BD)
	{
		const uint8_t f = pt[(uint32_t)cands[i] * G + g1];
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
	// Use counts as cursors while scattering, then restore them from starts.
	for (int i = tid; i < n; i += BD)
		members[atomicAdd(&starts[pat1[i]], 1)] = cands[i];
	__syncthreads();
	for (int i = tid; i < NPAT; i += BD) starts[i] -= counts[i];
	__syncthreads();

	int total = 0, by3 = 0; // only thread 0's copy is used
	if (counts[ALL_GREEN]) total = by3 = 1;
	uint32_t * mybits = bits + tid;

	for (int p = 0; p < ALL_GREEN; p++)
	{
		const int s = counts[p];
		if (s == 0) continue;
		// Groups of 1 or 2 are split perfectly by guessing one of their words.
		if (s < 3) { total += 4 * s - (s + 1); by3 += s; continue; }

		HardRule rule = base;
		if (HARD) rule.add(words + 5 * g1, p);
		const uint16_t * mem = members + starts[p];
		int best_d = 0, best_k = 0;
		for (int g2 = tid; g2 < G; g2 += BD)
		{
			if (*(volatile int *)&perfect) break;
			if (HARD && !rule.allows(words + 5 * g2)) continue;
#pragma unroll
			for (int w = 0; w < 8; w++) mybits[w * BD] = 0;
#pragma unroll 4
			for (int j = 0; j < s; j++)
			{
				const uint32_t q = pt[(uint32_t)mem[j] * G + g2];
				mybits[(q >> 5) * BD] |= 1u << (q & 31);
			}
			int d = 0;
#pragma unroll
			for (int w = 0; w < 8; w++) d += __popc(mybits[w * BD]);
			const int c = g2cand[g2];
			const int k = d + (c >= 0 && pat1[c] == p);
			best_d = max(best_d, d);
			best_k = max(best_k, k);
			if (k == s + 1) perfect = 1;
		}
		best_d = block_reduce(best_d, Max(), scratch);
		best_k = block_reduce(best_k, Max(), scratch);
		// Each word spends guesses 1 and 2; a group of s left after that needs at least
		// 2s-1 more, and the second guess itself, if it was the answer, needs none.
		total += 4 * s - best_k;
		by3 += best_d;
		__syncthreads();
		if (tid == 0) perfect = 0;
		__syncthreads();
	}
	if (tid == 0) out[blockIdx.x] = { total, by3 };
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
	dev_->pt.reserve((size_t)G * P);
	dev_->tt.reserve((size_t)G * P);
	table_kernel<<<dim3((G + BD - 1) / BD, P), BD>>>(dev_->words.p, dpool.p, G, P, dev_->pt.p, dev_->tt.p);
	CUDA_CHECK(cudaGetLastError());
	table_.resize((size_t)G * P);
	CUDA_CHECK(cudaMemcpy(table_.data(), dev_->tt.p, table_.size(), cudaMemcpyDeviceToHost));

	const size_t max_smem = (8 * BD + 512 + BD / 32) * 4 + 3 * (size_t)P;
	CUDA_CHECK(cudaFuncSetAttribute(lookahead_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_smem));
	CUDA_CHECK(cudaFuncSetAttribute(lookahead_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_smem));
}

Engine::~Engine()
{
	cudaEventDestroy(dev_->start);
	cudaEventDestroy(dev_->stop);
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

	std::vector<int> g2cand(G, -1);
	for (int i = 0; i < n; i++) g2cand[pool_[cands[i]]] = i;

	std::lock_guard<std::mutex> lock(dev_->mu);
	Device & d = *dev_;
	d.cands.upload(cands);
	d.guesses.upload(g1s);
	d.g2cand.upload(g2cand);
	d.look.reserve(g1s.size());
	const size_t smem = (8 * BD + 512 + BD / 32) * 4 + 3 * (size_t)n;
	const HardRule base = hard ? *hard : HardRule();

	// Grow the batch until a launch takes about LAUNCH_MS.
	size_t batch = 64;
	for (size_t at = 0; at < g1s.size();)
	{
		if (cancel && cancel->load()) return false;
		const size_t cnt = std::min(batch, g1s.size() - at);
		CUDA_CHECK(cudaEventRecord(d.start));
		if (hard)
			lookahead_kernel<true><<<cnt, BD, smem>>>(d.pt.p, G, d.words.p, d.cands.p, n,
				d.guesses.p + at, d.g2cand.p, base, d.look.p + at);
		else
			lookahead_kernel<false><<<cnt, BD, smem>>>(d.pt.p, G, d.words.p, d.cands.p, n,
				d.guesses.p + at, d.g2cand.p, base, d.look.p + at);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaEventRecord(d.stop));
		CUDA_CHECK(cudaEventSynchronize(d.stop));
		float ms = 0;
		CUDA_CHECK(cudaEventElapsedTime(&ms, d.start, d.stop));
		at += cnt;
		if (cnt == batch)
			batch = std::clamp<size_t>((size_t)(batch * LAUNCH_MS / std::max(ms, 0.01f)), 1, std::max<size_t>(batch * 4, 1));
		if (progress) progress((double)at / g1s.size());
	}
	CUDA_CHECK(cudaMemcpy(out.data(), d.look.p, out.size() * sizeof(Lookahead), cudaMemcpyDeviceToHost));
	return true;
}
