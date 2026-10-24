// Word lists, game state and the per-state analysis shared by the CLI and the GUI.
#pragma once

#include "engine.h"

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <vector>

struct WordLists
{
	std::vector<std::string> allowed; // every accepted guess
	std::vector<std::string> answers; // the original 2315 answers
};

// The lists compiled into the binary (data/allowed.txt, data/answers.txt).
WordLists builtin_lists();
// One word per line; throws on unreadable files or words that are not 5 letters a-z.
std::vector<std::string> read_list(const std::string & path);

// $XDG_CACHE_HOME/wordle-gpu or ~/.cache/wordle-gpu.
std::string default_cache_dir();

// Sorts and dedups lists.allowed; returns the indices of the possible answers in it
// (every word when all is set). Throws if an answer is not an allowed guess.
std::vector<int> prepare_pool(WordLists & lists, bool all);

// "gybbg", "21002" or "GY..G": g/2 green, y/1 yellow, b/x/./-/0 gray. Empty if malformed.
std::optional<uint8_t> parse_pattern(const std::string & s);
std::string pattern_string(uint8_t pat); // "gybbg"

struct Turn
{
	int guess;   // word index
	uint8_t pat;
};

constexpr int MAX_GUESSES = 6;

struct Suggestion
{
	int guess;
	GuessStats st;
	Lookahead la;    // valid when Analysis::have_lookahead
	int exact = -1;  // exact minimum total guesses over the candidates, -1 unknown, INF impossible
};

struct Analysis
{
	std::vector<uint16_t> cands; // pool indices still possible
	std::vector<Suggestion> rows;
	bool have_lookahead = false;
	bool proven = false;         // rows[0] is the exact optimum (by expected guesses)
	int cached = 0;              // exact values loaded from the opener cache
	int played = 0;              // guesses made before this one

	// Expected final score (guess number the game ends on) for a suggestion.
	double expected_lb(const Suggestion & s) const { return played + (double)s.la.total / cands.size(); }
	double expected_exact(const Suggestion & s) const { return played + (double)s.exact / cands.size(); }
};

class Game
{
public:
	Game(Engine & eng, bool hard) : eng_(eng), hard_(hard) {}

	Engine & engine() { return eng_; }
	const Engine & engine() const { return eng_; }
	bool hard() const { return hard_; }
	const std::vector<Turn> & turns() const { return turns_; }
	void push(int guess, uint8_t pat) { turns_.push_back({ guess, pat }); }
	void pop() { if (!turns_.empty()) turns_.pop_back(); }
	void clear() { turns_.clear(); }

	int find(const std::string & word) const; // word index or -1
	std::vector<uint16_t> candidates() const;
	HardRule rule() const;
	std::vector<int> guess_list() const; // all words, or the ones hard mode still allows

	enum Stage { STATS, LOOKAHEAD, EXACT };
	struct Options
	{
		bool lookahead = true;
		int lookahead_limit = 0;   // 0 = every guess, else only the best N by entropy
		int exact_top = 0;         // exact values for at least this many of the best guesses
		bool exact_prove = false;  // keep going until the best is proven optimal
		int threads = 0;           // exact solver threads, 0 = hardware
		std::string cache_dir;     // where opening results are kept; empty = no cache
	};
	// Called after each stage with the partial result; return value unused.
	using Report = std::function<void(Stage, const Analysis &)>;
	using Progress = std::function<void(Stage, double)>;
	// Returns false if cancelled (the last reported Analysis is still valid).
	bool analyze(const Options & opt, Analysis & out, const Report & report = {},
		const Progress & progress = {}, const std::atomic<bool> * cancel = nullptr);

private:
	Engine & eng_;
	bool hard_;
	std::vector<Turn> turns_;
};

// Ranking used everywhere: exact when known for both, else the two-ply bound,
// then prefer guesses that can win, then entropy.
bool better(const Analysis & a, const Suggestion & x, const Suggestion & y);
