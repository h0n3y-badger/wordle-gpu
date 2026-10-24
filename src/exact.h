// Exact minimum expected guesses by depth-first branch and bound on the CPU,
// over every allowed guess at every node.
#pragma once

#include "engine.h"

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

constexpr int INF = 1 << 24;

class ExactSolver
{
public:
	ExactSolver(const Engine & eng, bool hard, int threads = 0);
	~ExactSolver();

	// For each first guess g, the minimum total number of guesses summed over all
	// candidates (each needing at most `left` guesses, g included), or INF.
	// Returns false if cancelled.
	bool totals(const std::vector<uint16_t> & cands, const std::vector<int> & g1s, int left,
		const HardRule & rule, std::vector<int> & out,
		const std::function<void(double)> & progress = {}, const std::atomic<bool> * cancel = nullptr);

	// Minimum total guesses for the candidate set, < beta, or some value >= beta.
	int solve(const uint16_t * s, int n, int left, const HardRule & rule, int beta);

private:
	struct Memo;
	const Engine & eng_;
	bool hard_;
	int threads_;
	std::unique_ptr<Memo> memo_;
	const std::atomic<bool> * cancel_ = nullptr;
};
