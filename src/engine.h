// GPU side: feedback tables and the brute-force scoring kernels.
#pragma once

#include "feedback.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// How one guess splits the remaining candidates.
struct GuessStats
{
	int buckets;   // distinct feedback patterns
	int largest;   // worst-case number of candidates left
	int sum_sq;    // sum of bucket sizes squared; expected candidates left = sum_sq / n
	float entropy; // bits
	int candidate; // the guess is itself a remaining candidate
};

// Two-ply brute force: every feedback of this guess, followed by every possible second guess.
struct Lookahead
{
	// Lower bound on the total guesses summed over all candidates, counting this guess.
	// Exact for the first two guesses; after that a group of s words is charged 2s-1,
	// the best any strategy can do.
	int total;
	// Candidates solved by the third guess from here (this guess + 2), playing the best
	// second guess for each feedback and then a word from the group.
	int by3;
};

class Engine
{
public:
	// words: every allowed guess. pool: indices into words of the possible answers.
	Engine(std::vector<std::string> words, std::vector<int> pool, int device = 0);
	~Engine();

	const std::string & device_name() const { return device_name_; }
	const std::vector<std::string> & words() const { return words_; }
	const std::vector<int> & pool() const { return pool_; }
	int num_guesses() const { return (int)words_.size(); }
	int pool_size() const { return (int)pool_.size(); }

	// Feedback for guess g against pool entry p, from the host copy of the table.
	uint8_t pat(int g, int p) const { return table_[(size_t)g * pool_.size() + p]; }
	const uint8_t * row(int g) const { return table_.data() + (size_t)g * pool_.size(); }
	// The same table pool-major: row p at p * stride, guesses padded with zeros to a
	// multiple of 64 and 64-byte aligned. Copied from the GPU on first use.
	const uint8_t * pool_major(size_t & stride) const;

	// cands are pool indices; guesses are word indices. One result per guess.
	void stats(const std::vector<uint16_t> & cands, const std::vector<int> & guesses,
		std::vector<GuessStats> & out);

	// One result per first guess in g1s. Second guesses range over all words, or the
	// ones hard mode allows after each feedback when hard is set. Returns false if
	// cancelled. Work is split into launches of a few tens of ms so the desktop stays smooth.
	// group_k, if given, gets NPAT values per first guess: for each feedback group of 3+
	// candidates the best k (see Lookahead), so that group needs 3s-k more guesses; 0 for
	// smaller groups.
	bool lookahead(const std::vector<uint16_t> & cands, const std::vector<int> & g1s,
		const HardRule * hard, std::vector<Lookahead> & out,
		const std::function<void(double)> & progress = {}, const std::atomic<bool> * cancel = nullptr,
		std::vector<int> * group_k = nullptr);

private:
	struct Device;
	std::unique_ptr<Device> dev_;
	std::vector<std::string> words_;
	std::vector<int> pool_;
	std::vector<uint8_t> table_; // guess-major
	mutable std::once_flag pm_once_;
	mutable std::unique_ptr<uint8_t, void (*)(void *)> pm_{ nullptr, std::free };
	mutable size_t pm_stride_ = 0;
	std::string device_name_;
};
