#include "exact.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

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

// Nodes with at least this many candidates share their options with idle threads.
constexpr int SHARE_MIN = 24;

std::string make_key(const uint16_t * s, int n, int left, const HardRule * rule)
{
	std::string k((const char *)s, n * sizeof(uint16_t));
	k.push_back((char)left);
	if (rule) k.append((const char *)rule, sizeof(HardRule));
	return k;
}

}

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
	  memo_(new Memo)
{
}

ExactSolver::~ExactSolver() = default;

int ExactSolver::solve(const uint16_t * s, int n, int left, const HardRule & rule, int beta)
{
	if (left <= 0) return INF;
	if (n == 1) return 1;
	if (left == 1) return INF;
	if (n == 2) return 3;
	if (cancel_ && cancel_->load(std::memory_order_relaxed)) throw Cancelled();

	const std::string key = make_key(s, n, left, hard_ ? &rule : nullptr);
	Memo::Entry e;
	if (memo_->get(key, e) && (e.exact || e.value >= beta)) return e.value;

	const auto & pool = eng_.pool();
	const auto & words = eng_.words();
	Hist & h = tl_hist;

	// Guessing a candidate that tells all the others apart is optimal: 2n-1.
	for (int i = 0; i < n; i++)
	{
		const uint8_t * row = eng_.row(pool[s[i]]);
		h.reset();
		for (int j = 0; j < n; j++) h.add(row[s[j]]);
		if (h.nused == n)
		{
			memo_->put(key, { 2 * n - 1, true });
			return 2 * n - 1;
		}
	}

	std::vector<Opt> opts;
	const int G = eng_.num_guesses();
	for (int g = 0; g < G; g++)
	{
		if (hard_ && !rule.allows(words[g].data())) continue;
		const uint8_t * row = eng_.row(g);
		h.reset();
		for (int j = 0; j < n; j++) h.add(row[s[j]]);
		const int in = h.get(ALL_GREEN);
		const int groups = h.nused - in;
		if (!in && groups == 1) continue; // learns nothing
		// Every word spends this guess; a group of k needs at least 2k-1 more.
		const int lb = n + 2 * (n - in) - groups;
		if (lb < beta) opts.push_back({ lb, g });
	}
	std::sort(opts.begin(), opts.end(), [](const Opt & a, const Opt & b) { return a.lb < b.lb; });

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
	std::vector<std::pair<int, int>> groups; // (size, pattern)
	for (int k = 0; k < h.nused; k++)
		if (h.used[k] != ALL_GREEN) groups.push_back({ h.cnt[h.used[k]], h.used[k] });
	// Largest groups first: they decide whether the bound is beaten.
	std::sort(groups.begin(), groups.end(), std::greater<>());
	int start[NPAT];
	int at = 0;
	for (auto [size, pat] : groups) { start[pat] = at; at += size; }
	std::vector<uint16_t> part(at);
	for (int j = 0; j < n; j++)
	{
		const uint8_t f = row[s[j]];
		if (f != ALL_GREEN) part[start[f]++] = s[j];
	}

	int total = n, rest = o.lb - n;
	at = 0;
	for (auto [size, pat] : groups)
	{
		rest -= 2 * size - 1;
		HardRule sub = rule;
		if (hard_) sub.add(eng_.words()[o.g].data(), pat);
		total += solve(part.data() + at, size, left - 1, sub, best - total - rest);
		at += size;
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
