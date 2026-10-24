// Exact minimum expected guesses by depth-first branch and bound on the CPU,
// over every allowed guess at every node.
#pragma once

#include "engine.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

constexpr int INF = 1 << 24;

class ExactSolver
{
public:
	ExactSolver(const Engine & eng, bool hard, int threads = 0);
	~ExactSolver();

	// For each first guess g, the minimum total number of guesses summed over all
	// candidates (each needing at most `left` guesses, g included), or INF.
	// Returns false if cancelled; guesses finished before that keep their value, the rest get -1.
	// With beta (one per guess), a guess may stop as soon as its total is known to be at
	// least its beta: out is then exact if below beta, else a lower bound >= beta. group_k
	// (one pointer per guess to Engine::lookahead's NPAT values, or null) gives each group
	// its two-ply bound to start from; without it groups start at 2s-1, which prunes far less.
	bool totals(const std::vector<uint16_t> & cands, const std::vector<int> & g1s, int left,
		const HardRule & rule, std::vector<int> & out,
		const std::function<void(double)> & progress = {}, const std::atomic<bool> * cancel = nullptr,
		const std::vector<int> * beta = nullptr, const std::vector<const int *> * group_k = nullptr);

	// Minimum total guesses for the candidate set, < beta, or some value >= beta.
	int solve(const uint16_t * s, int n, int left, const HardRule & rule, int beta);

	// A guess (word index) that achieves the set's minimum total, or -1 if the set can't
	// be finished within `left` guesses. Lowest bound first, so usually the first try.
	int choose(const uint16_t * s, int n, int left, const HardRule & rule);

private:
	struct Memo;
	struct Tables;
	struct Opt { int lb, g; };
	// A node whose options any idle thread may help evaluate.
	struct Shared
	{
		const uint16_t * s;
		int n, left;
		const HardRule * rule;
		const std::vector<Opt> * opts;
		std::atomic<size_t> next{ 0 };
		std::atomic<int> best;
		int helpers = 0; // guarded by shared_mu_
		int caps = -1;
	};

	// Total for guessing o.g at this node, or `best` if it can't beat best. caps is
	// km << 8 | kn for the node's CapBound, or -1.
	int try_option(const uint16_t * s, int n, int left, const HardRule & rule, const Opt & o, int best, int caps);
	void work_on(Shared & sh);
	bool help(); // false if no node needs help

	std::mutex shared_mu_;
	std::condition_variable shared_cv_;
	std::vector<Shared *> shared_;
	const Engine & eng_;
	bool hard_;
	int threads_;
	std::unique_ptr<Memo> memo_;
	std::unique_ptr<Tables> tab_;
	const std::atomic<bool> * cancel_ = nullptr;
};
