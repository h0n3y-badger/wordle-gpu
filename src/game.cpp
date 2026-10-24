#include "game.h"
#include "exact.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

extern const char WORDS_ALLOWED[];
extern const char WORDS_ANSWERS[];

static bool valid_word(const std::string & w)
{
	return w.size() == 5 && std::all_of(w.begin(), w.end(), [](char c) { return c >= 'a' && c <= 'z'; });
}

static std::vector<std::string> parse_list(std::istream & in, const std::string & name)
{
	std::vector<std::string> out;
	for (std::string line; std::getline(in, line);)
	{
		while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
		if (line.empty()) continue;
		for (char & c : line) c = (char)std::tolower((unsigned char)c);
		if (!valid_word(line)) throw std::runtime_error(name + ": not a 5-letter word: " + line);
		out.push_back(line);
	}
	return out;
}

WordLists builtin_lists()
{
	std::istringstream a(WORDS_ALLOWED), b(WORDS_ANSWERS);
	return { parse_list(a, "allowed"), parse_list(b, "answers") };
}

std::vector<std::string> read_list(const std::string & path)
{
	std::ifstream in(path);
	if (!in) throw std::runtime_error("cannot read " + path);
	return parse_list(in, path);
}

std::vector<int> prepare_pool(WordLists & lists, bool all)
{
	auto & w = lists.allowed;
	std::sort(w.begin(), w.end());
	w.erase(std::unique(w.begin(), w.end()), w.end());
	std::vector<int> pool;
	if (all)
		for (size_t i = 0; i < w.size(); i++) pool.push_back((int)i);
	else
		for (const auto & a : lists.answers)
		{
			auto it = std::lower_bound(w.begin(), w.end(), a);
			if (it == w.end() || *it != a) throw std::runtime_error("answer not in the allowed list: " + a);
			pool.push_back((int)(it - w.begin()));
		}
	std::sort(pool.begin(), pool.end());
	pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
	return pool;
}

std::optional<uint8_t> parse_pattern(const std::string & s)
{
	if (s.size() != 5) return std::nullopt;
	int pat = 0;
	for (int i = 4; i >= 0; i--)
	{
		int d;
		switch (std::tolower((unsigned char)s[i]))
		{
			case 'g': case '2': d = 2; break;
			case 'y': case '1': d = 1; break;
			case 'b': case 'x': case '.': case '-': case '0': d = 0; break;
			default: return std::nullopt;
		}
		pat = pat * 3 + d;
	}
	return (uint8_t)pat;
}

std::string pattern_string(uint8_t pat)
{
	std::string s(5, 'b');
	for (int i = 0; i < 5; i++) s[i] = "byg"[pat_digit(pat, i)];
	return s;
}

int Game::find(const std::string & word) const
{
	const auto & w = eng_.words();
	auto it = std::lower_bound(w.begin(), w.end(), word);
	if (it != w.end() && *it == word) return (int)(it - w.begin());
	// Lists read from files need not be sorted.
	it = std::find(w.begin(), w.end(), word);
	return it == w.end() ? -1 : (int)(it - w.begin());
}

std::vector<uint16_t> Game::candidates() const
{
	std::vector<uint16_t> out;
	for (int p = 0; p < eng_.pool_size(); p++)
	{
		bool ok = true;
		for (const Turn & t : turns_) ok = ok && eng_.pat(t.guess, p) == t.pat;
		if (ok) out.push_back((uint16_t)p);
	}
	return out;
}

HardRule Game::rule() const
{
	HardRule r;
	for (const Turn & t : turns_) r.add(eng_.words()[t.guess].data(), t.pat);
	return r;
}

std::vector<int> Game::guess_list() const
{
	const HardRule r = rule();
	std::vector<int> out;
	for (int g = 0; g < eng_.num_guesses(); g++)
		if (!hard_ || r.allows(eng_.words()[g].data())) out.push_back(g);
	return out;
}

static int key(const Suggestion & s) { return s.exact >= 0 ? s.exact : s.la.total; }

bool better(const Analysis & a, const Suggestion & x, const Suggestion & y)
{
	if (a.have_lookahead)
	{
		if (key(x) != key(y)) return key(x) < key(y);
		if (x.st.candidate != y.st.candidate) return x.st.candidate > y.st.candidate;
		if (x.la.by3 != y.la.by3) return x.la.by3 > y.la.by3;
	}
	else if (x.st.candidate != y.st.candidate && x.st.buckets == y.st.buckets)
		return x.st.candidate > y.st.candidate;
	if (x.st.entropy != y.st.entropy) return x.st.entropy > y.st.entropy;
	return x.guess < y.guess;
}

static void sort_rows(Analysis & a)
{
	std::stable_sort(a.rows.begin(), a.rows.end(),
		[&](const Suggestion & x, const Suggestion & y) { return better(a, x, y); });
}

bool Game::analyze(const Options & opt, Analysis & out, const Report & report,
	const Progress & progress, const std::atomic<bool> * cancel)
{
	out = Analysis();
	out.played = (int)turns_.size();
	out.cands = candidates();
	const int n = (int)out.cands.size();
	const std::vector<int> guesses = guess_list();

	std::vector<GuessStats> st;
	eng_.stats(out.cands, guesses, st);
	out.rows.resize(guesses.size());
	for (size_t i = 0; i < guesses.size(); i++)
	{
		Suggestion & s = out.rows[i];
		s.guess = guesses[i];
		s.st = st[i];
		// Until the lookahead runs: every word spends this guess and a group of k needs 2k-1 more.
		const int groups = st[i].buckets - st[i].candidate;
		s.la = { n + 2 * (n - st[i].candidate) - groups, -1 };
	}
	sort_rows(out);
	if (report) report(STATS, out);
	if (n == 0 || out.rows.empty() || out.played >= MAX_GUESSES) return true;
	auto stage_progress = [&](Stage s) {
		return [&progress, s](double f) { if (progress) progress(s, f); };
	};

	if (opt.lookahead)
	{
		size_t m = out.rows.size();
		if (opt.lookahead_limit > 0) m = std::min(m, (size_t)opt.lookahead_limit);
		std::vector<int> g1s(m);
		for (size_t i = 0; i < m; i++) g1s[i] = out.rows[i].guess;
		const HardRule r = rule();
		std::vector<Lookahead> la;
		if (!eng_.lookahead(out.cands, g1s, hard_ ? &r : nullptr, la, stage_progress(LOOKAHEAD), cancel))
			return false;
		for (size_t i = 0; i < m; i++) out.rows[i].la = la[i];
		out.have_lookahead = true;
		sort_rows(out);
		if (report) report(LOOKAHEAD, out);
	}

	if (opt.exact_top > 0 || opt.exact_prove)
	{
		ExactSolver solver(eng_, hard_, opt.threads);
		const int left = MAX_GUESSES - out.played;
		const HardRule r = rule();
		const size_t chunk = std::max<size_t>(8, std::thread::hardware_concurrency());
		const size_t top = opt.exact_top;
		size_t done = 0;
		int best = INF;
		while (done < out.rows.size())
		{
			size_t want = done < top ? top : done + chunk;
			want = std::min(want, out.rows.size());
			std::vector<int> g1s;
			for (size_t i = done; i < want; i++) g1s.push_back(out.rows[i].guess);
			std::vector<int> ex;
			auto prog = [&](double f) {
				if (progress) progress(EXACT, std::min(1.0, (done + f * g1s.size()) / want));
			};
			if (!solver.totals(out.cands, g1s, left, r, ex, prog, cancel)) return false;
			for (size_t i = 0; i < g1s.size(); i++)
			{
				out.rows[done + i].exact = ex[i];
				best = std::min(best, ex[i]);
			}
			done = want;
			// Only the solved rows are reordered, so the rest stays sorted by lower bound.
			std::stable_sort(out.rows.begin(), out.rows.begin() + done,
				[&](const Suggestion & x, const Suggestion & y) { return better(out, x, y); });
			out.proven = best < INF && std::all_of(out.rows.begin() + done, out.rows.end(),
				[&](const Suggestion & s) { return s.la.total >= best; });
			if (report) report(EXACT, out);
			if (done >= top && (out.proven || !opt.exact_prove)) break;
		}
		sort_rows(out);
		if (report) report(EXACT, out);
	}
	return true;
}
