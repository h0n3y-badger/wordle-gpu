#include "exact.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#ifdef __AVX512BW__
#include <immintrin.h>
#endif

namespace {

struct Cancelled {};

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

// Sets up to this size count distinct feedbacks 64 guesses at a time by comparing
// every pair of candidates (counts are bytes, so at most 255).
constexpr int VEC_MAX = 192;

struct RawOpt { int lb, g; };

// Per-thread scratch for the guess scan; nothing in it lives across a recursive call.
struct Scratch
{
#ifdef __AVX512BW__
	__m512i buf[VEC_MAX];
#endif
	const uint8_t * rows[VEC_MAX];
	std::vector<uint64_t> allow, cand; // one bit per guess
	std::vector<RawOpt> raw;
	std::vector<int> bucket;
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
// Distinct feedbacks over the n candidate rows for 64 neighbouring guesses, one byte per guess:
// a row adds one wherever it differs from every earlier row. Four rows go at a time so the
// four masked-compare chains overlap.
inline __m512i distinct64(const uint8_t * const * rows, size_t off, int n, __m512i * buf)
{
	const __m512i one = _mm512_set1_epi8(1);
	__m512i cnt = one;
	buf[0] = _mm512_load_si512(rows[0] + off);
	int j = 1;
	for (; j + 3 < n; j += 4)
	{
		const __m512i a = _mm512_load_si512(rows[j] + off), b = _mm512_load_si512(rows[j + 1] + off);
		const __m512i c = _mm512_load_si512(rows[j + 2] + off), d = _mm512_load_si512(rows[j + 3] + off);
		__mmask64 ka = ~0ull;
		__mmask64 kb = _mm512_cmpneq_epu8_mask(b, a);
		__mmask64 kc = _mm512_mask_cmpneq_epu8_mask(_mm512_cmpneq_epu8_mask(c, a), c, b);
		__mmask64 kd = _mm512_mask_cmpneq_epu8_mask(_mm512_mask_cmpneq_epu8_mask(_mm512_cmpneq_epu8_mask(d, a), d, b), d, c);
		for (int i = 0; i < j; i++)
		{
			const __m512i x = buf[i];
			ka = _mm512_mask_cmpneq_epu8_mask(ka, a, x);
			kb = _mm512_mask_cmpneq_epu8_mask(kb, b, x);
			kc = _mm512_mask_cmpneq_epu8_mask(kc, c, x);
			kd = _mm512_mask_cmpneq_epu8_mask(kd, d, x);
		}
		buf[j] = a;
		buf[j + 1] = b;
		buf[j + 2] = c;
		buf[j + 3] = d;
		cnt = _mm512_mask_add_epi8(cnt, ka, cnt, one);
		cnt = _mm512_mask_add_epi8(cnt, kb, cnt, one);
		cnt = _mm512_mask_add_epi8(cnt, kc, cnt, one);
		cnt = _mm512_mask_add_epi8(cnt, kd, cnt, one);
	}
	for (; j < n; j++)
	{
		const __m512i a = _mm512_load_si512(rows[j] + off);
		__mmask64 ka = ~0ull;
		for (int i = 0; i < j; i++) ka = _mm512_mask_cmpneq_epu8_mask(ka, a, buf[i]);
		buf[j] = a;
		cnt = _mm512_mask_add_epi8(cnt, ka, cnt, one);
	}
	return cnt;
}
#endif

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

	// Scan every allowed guess for its bound n + 2(n-in) - groups = 3n - in - distinct.
	// A bound of 2n is met exactly by a non-candidate splitting all apart, or (with a
	// guess to spare) a candidate leaving one pair: that settles the node at once.
	// With two guesses left nothing else can finish, so only those are looked for.
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
	sc.raw.clear();
	// Guesses with more distinct feedbacks than this may have bound < beta (candidates
	// one less), and each must have at least two.
	const int lo = left == 2 ? n - 1 : std::max(1, 3 * n - 1 - beta);
	bool settled = false;
	auto take = [&](int g, int distinct) {
		const int in = (int)(sc.cand[g >> 6] >> (g & 63) & 1);
		const int lb = 3 * n - in - distinct;
		if (lb == 2 * n && (!in || left >= 3)) settled = true;
		else if (left >= 3 && lb < beta) sc.raw.push_back({ lb, g });
	};
#ifdef __AVX512BW__
	if (n <= VEC_MAX)
	{
		for (int j = 0; j < n; j++) sc.rows[j] = t.prow(s[j]);
		const __m512i vlo = _mm512_set1_epi8((char)lo);
		alignas(64) uint8_t c8[64];
		for (int b = 0; b < W && !settled; b++)
		{
			if (!allow[b]) continue;
			const __m512i cnt = distinct64(sc.rows, (size_t)b * 64, n, sc.buf);
			uint64_t sel = _mm512_cmpgt_epu8_mask(cnt, vlo) & allow[b];
			if (!sel) continue;
			_mm512_store_si512(c8, cnt);
			for (; sel && !settled; sel &= sel - 1)
			{
				const int l = std::countr_zero(sel);
				take(b * 64 + l, c8[l]);
			}
		}
	}
	else
#endif
	{
		for (int b = 0; b < W && !settled; b++)
			for (uint64_t m = allow[b]; m && !settled; m &= m - 1)
			{
				const int g = b * 64 + std::countr_zero(m);
				const uint8_t * row = eng_.row(g);
				uint64_t seen[4] = { 0, 0, 0, 0 };
				for (int j = 0; j < n; j++)
				{
					const uint8_t f = row[s[j]];
					seen[f >> 6] |= 1ull << (f & 63);
				}
				const int distinct = std::popcount(seen[0]) + std::popcount(seen[1]) + std::popcount(seen[2]) + std::popcount(seen[3]);
				if (distinct > lo) take(g, distinct);
			}
	}
	if (settled)
	{
		memo_->put(key, { 2 * n, true });
		return 2 * n;
	}
	if (left == 2)
	{
		memo_->put(key, { INF, true });
		return INF;
	}

	// Counting sort by bound, then non-candidates before candidates (one more group, so
	// smaller ones), then guess order. Bounds lie in [2n-1, 3n-2].
	const int lb0 = 2 * n - 1;
	auto slot = [&](const RawOpt & o) { return 2 * (o.lb - lb0) + (int)(sc.cand[o.g >> 6] >> (o.g & 63) & 1); };
	sc.bucket.assign(2 * n + 2, 0);
	for (const RawOpt & o : sc.raw) sc.bucket[slot(o)]++;
	for (int i = 0, sum = 0; i < 2 * n + 2; i++) { const int c = sc.bucket[i]; sc.bucket[i] = sum; sum += c; }
	std::vector<Opt> opts(sc.raw.size());
	for (const RawOpt & o : sc.raw) opts[sc.bucket[slot(o)]++] = { o.lb, o.g };

	int best = beta;
	if (n < SHARE_MIN || threads_ == 1)
	{
		for (const Opt & o : opts)
		{
			if (o.lb >= best) break;
			best = std::min(best, try_option(s, n, left, rule, o, best));
		}
	}
	else
	{
		// Big node: let idle threads take options too, all pruning against one best.
		Shared sh{ s, n, left, &rule, &opts };
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

int ExactSolver::try_option(const uint16_t * s, int n, int left, const HardRule & rule, const Opt & o, int best)
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
	// any other at least 2s. That alone refutes many options.
	int open[NPAT], nopen = 0;
	for (int i = 0; i < ng; i++)
	{
		const int size = groups[i] >> 8;
		if (perfect_split(eng_, part + off[i], size)) { total += 2 * size - 1; rest -= 2 * size - 1; }
		else
		{
			open[nopen++] = i;
			if (total + ++rest >= best) return best;
		}
	}
	for (int k = 0; k < nopen; k++)
	{
		const int i = open[k];
		const int size = groups[i] >> 8;
		rest -= 2 * size;
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
		const int t = try_option(sh.s, sh.n, sh.left, *sh.rule, o, best);
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
	const std::function<void(double)> & progress, const std::atomic<bool> * cancel)
{
	cancel_ = cancel;
	const int n = (int)cands.size();
	out.assign(g1s.size(), n);

	struct Task { int idx; std::vector<uint16_t> set; HardRule rule; };
	std::vector<Task> tasks;
	for (size_t i = 0; i < g1s.size(); i++)
	{
		const int g = g1s[i];
		const uint8_t * row = eng_.row(g);
		std::vector<std::vector<uint16_t>> parts(NPAT);
		for (uint16_t c : cands) parts[row[c]].push_back(c);
		for (int p = 0; p < ALL_GREEN; p++)
		{
			if (parts[p].empty()) continue;
			HardRule sub = rule;
			if (hard_) sub.add(eng_.words()[g].data(), p);
			tasks.push_back({ (int)i, std::move(parts[p]), sub });
		}
	}
	// Big groups first so the threads finish together.
	std::sort(tasks.begin(), tasks.end(), [](const Task & a, const Task & b) { return a.set.size() > b.set.size(); });

	std::atomic<size_t> next{ 0 }, done{ 0 };
	std::atomic<bool> cancelled{ false };
	std::vector<int> results(tasks.size(), -1);
	std::mutex progress_mu;
	auto work = [&] {
		try
		{
			for (;;)
			{
				const size_t t = next++;
				if (t < tasks.size())
				{
					results[t] = solve(tasks[t].set.data(), (int)tasks[t].set.size(), left - 1, tasks[t].rule, INF);
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

	// After a cancel, guesses with every group solved still get their value.
	for (size_t t = 0; t < tasks.size(); t++)
	{
		int & o = out[tasks[t].idx];
		if (o >= 0) o = results[t] < 0 ? -1 : std::min(INF, o + results[t]);
	}
	return !cancelled;
}
