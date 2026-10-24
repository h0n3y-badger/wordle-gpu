// wordle-gpu-solve: brute-force the whole game. Scores every opening guess on the GPU,
// then solves openers exactly, most promising first, until all are done, the time
// limit hits or Ctrl-C; then writes every opener ranked by average score and, for the
// best ones, every game the optimal strategy plays.
#include "game.h"
#include "exact.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <mutex>
#include <thread>
#include <unistd.h>

static void usage(const char * argv0)
{
	fprintf(stderr,
		"usage: %s [options] [guess=pattern ...]\n"
		"Solves every opening guess exactly, best-looking first, until done, -T or Ctrl-C,\n"
		"then writes PREFIX-openers.txt (all openers ranked by average score) and\n"
		"PREFIX-games.txt (every game of the optimal strategy for the best -k openers).\n"
		"Turns given as word=pattern (g/y/b) start from that position instead.\n"
		"  -T TIME     time limit, e.g. 90, 30m, 2h (default: until done or Ctrl-C)\n"
		"  -k N|all    write the games of the best N solved openers (default 10)\n"
		"  -O PREFIX   output file prefix (default wordle-solve-<answers>[-hard])\n"
		"  -H          hard mode\n"
		"  -o          only the original 2315 answers (default: original + NYT picks)\n"
		"  -a          any allowed word can be the answer\n"
		"  -t N        threads (default all)\n"
		"  -d N        CUDA device (default 0)\n", argv0);
}

static double parse_duration(const std::string & s)
{
	char * end;
	const double v = strtod(s.c_str(), &end);
	const std::string unit = end;
	if (end == s.c_str() || v <= 0) return -1;
	if (unit == "" || unit == "s") return v;
	if (unit == "m") return v * 60;
	if (unit == "h") return v * 3600;
	return -1;
}

static std::atomic<bool> g_stop{ false };
static void on_sigint(int) { g_stop = true; }

struct Opener
{
	int guess;
	GuessStats st;
	int bound;       // two-ply lower bound on the total
	int total = -1;  // exact total, -1 unsolved, INF impossible within the guesses left
};

// One answer's game under the strategy: the guesses and the colours each got.
struct Game1
{
	int answer;
	std::vector<std::pair<int, uint8_t>> turns;
};

// Plays the optimal strategy (first guess fixed) against every candidate.
static void play_all(ExactSolver & solver, const Engine & eng, bool hard, std::vector<uint16_t> set,
	int left, HardRule rule, int forced, std::vector<std::pair<int, uint8_t>> & path, std::vector<Game1> & out)
{
	const int g = forced >= 0 ? forced : solver.choose(set.data(), (int)set.size(), left, rule);
	if (g < 0) throw std::runtime_error("no strategy finishes within the guess limit");
	std::vector<std::vector<uint16_t>> groups(NPAT);
	for (uint16_t c : set) groups[eng.pat(g, c)].push_back(c);
	for (int p = 0; p < NPAT; p++)
	{
		if (groups[p].empty()) continue;
		path.push_back({ g, (uint8_t)p });
		if (p == ALL_GREEN)
			out.push_back({ eng.pool()[groups[p][0]], path });
		else
		{
			HardRule sub = rule;
			if (hard) sub.add(eng.words()[g].data(), p);
			play_all(solver, eng, hard, std::move(groups[p]), left - 1, sub, -1, path, out);
		}
		path.pop_back();
	}
}

int main(int argc, char ** argv)
{
	double budget = 0;
	int top_k = 10, threads = 0, device = 0;
	bool hard = false;
	Pool pool_kind = Pool::NYT;
	std::string prefix;
	for (int c; (c = getopt(argc, argv, "T:k:O:Hoat:d:h")) != -1;)
	{
		switch (c)
		{
			case 'T':
				budget = parse_duration(optarg);
				if (budget <= 0) { fprintf(stderr, "bad time '%s' (e.g. 90, 30m, 2h)\n", optarg); return 2; }
				break;
			case 'k': top_k = std::string(optarg) == "all" ? -1 : atoi(optarg); break;
			case 'O': prefix = optarg; break;
			case 'H': hard = true; break;
			case 'o': pool_kind = Pool::ORIGINAL; break;
			case 'a': pool_kind = Pool::ALL; break;
			case 't': threads = atoi(optarg); break;
			case 'd': device = atoi(optarg); break;
			default: usage(argv[0]); return c == 'h' ? 0 : 2;
		}
	}
	if (prefix.empty())
		prefix = std::string("wordle-solve-") + (pool_kind == Pool::NYT ? "nyt" : pool_kind == Pool::ORIGINAL ? "original" : "any")
			+ (hard ? "-hard" : "");
	if (threads <= 0) threads = (int)std::max(1u, std::thread::hardware_concurrency());

	try
	{
		WordLists lists = builtin_lists();
		const std::vector<int> pool = prepare_pool(lists, pool_kind);
		const auto t0 = std::chrono::steady_clock::now();
		auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
		Engine eng(lists.allowed, pool, device);
		Game game(eng, hard);
		for (int i = optind; i < argc; i++)
		{
			const std::string arg = argv[i];
			const int g = arg.size() == 11 && arg[5] == '=' ? game.find(arg.substr(0, 5)) : -1;
			const auto pat = g >= 0 ? parse_pattern(arg.substr(6)) : std::nullopt;
			if (!pat) { fprintf(stderr, "bad turn '%s' (want word=pattern)\n", arg.c_str()); return 2; }
			game.push(g, *pat);
		}
		const std::vector<uint16_t> cands = game.candidates();
		const std::vector<int> guesses = game.guess_list();
		const HardRule rule = game.rule();
		const int n = (int)cands.size(), played = (int)game.turns().size(), left = MAX_GUESSES - played;
		if (n == 0 || left <= 0) { fprintf(stderr, "no possible answers or no guesses left\n"); return 1; }
		fprintf(stderr, "%s: %d answers, %d guesses, %s mode\n", eng.device_name().c_str(), n,
			(int)guesses.size(), hard ? "hard" : "normal");

		// Stop at the time limit or on Ctrl-C; a second Ctrl-C while writing quits.
		std::mutex mu;
		std::condition_variable cv;
		bool searching = true;
		std::thread timer;
		if (budget > 0)
			timer = std::thread([&] {
				std::unique_lock<std::mutex> lock(mu);
				if (!cv.wait_for(lock, std::chrono::duration<double>(budget), [&] { return !searching; }))
					g_stop = true;
			});
		signal(SIGINT, on_sigint);
		const bool tty = isatty(2);

		// GPU: stats and the two-ply brute force for every opener.
		std::vector<GuessStats> st;
		eng.stats(cands, guesses, st);
		std::vector<Lookahead> la;
		std::vector<int> group_k;
		double shown = -1;
		const bool looked = eng.lookahead(cands, guesses, hard ? &rule : nullptr, la,
			[&](double f) {
				if (tty && secs() - shown > 0.2) { shown = secs(); fprintf(stderr, "\rGPU brute force %5.1f%%", f * 100); }
			}, &g_stop, &group_k);
		const double gpu_secs = secs();
		if (tty) fprintf(stderr, "\r%40s\r", "");
		fprintf(stderr, "GPU brute force: %zu openers x %d second guesses x %d answers in %.2f s\n",
			guesses.size(), eng.num_guesses(), n, gpu_secs);
		if (!looked) { fprintf(stderr, "stopped during the GPU brute force; nothing to write\n"); return 1; }

		std::vector<Opener> openers(guesses.size());
		for (size_t i = 0; i < guesses.size(); i++) openers[i] = { guesses[i], st[i], la[i].total };
		std::vector<size_t> order(openers.size());
		for (size_t i = 0; i < order.size(); i++) order[i] = i;
		std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
			if (openers[a].bound != openers[b].bound) return openers[a].bound < openers[b].bound;
			return openers[a].st.entropy > openers[b].st.entropy;
		});

		// Exact: solve openers in bound order, a batch per round.
		ExactSolver solver(eng, hard, threads);
		size_t next = 0, solved = 0;
		int best = INF, best_guess = -1;
		const size_t batch = (size_t)threads;
		while (next < order.size() && !g_stop)
		{
			std::vector<int> g1s, out;
			std::vector<const int *> gk;
			std::vector<size_t> idx;
			for (; next < order.size() && idx.size() < batch; next++)
			{
				const size_t i = order[next];
				idx.push_back(i);
				g1s.push_back(openers[i].guess);
				gk.push_back(&group_k[i * NPAT]);
			}
			const size_t base = solved;
			solver.totals(cands, g1s, left, rule, out,
				[&](double f) {
					if (tty && secs() - shown > 0.25)
					{
						shown = secs();
						fprintf(stderr, "\rexact: %zu/%zu openers, best %s %.4f, %.0f s   ",
							base + (size_t)(f * idx.size()), order.size(),
							best_guess >= 0 ? eng.words()[best_guess].c_str() : "-",
							best < INF ? played + (double)best / n : 0.0, secs());
					}
				}, &g_stop, nullptr, &gk);
			for (size_t k = 0; k < idx.size(); k++)
			{
				if (out[k] < 0) continue;
				openers[idx[k]].total = out[k];
				solved++;
				if (out[k] < best) { best = out[k]; best_guess = g1s[k]; }
			}
		}
		{
			std::lock_guard<std::mutex> lock(mu);
			searching = false;
		}
		cv.notify_all();
		if (timer.joinable()) timer.join();
		signal(SIGINT, SIG_DFL);
		const double search_secs = secs();
		if (tty) fprintf(stderr, "\r%70s\r", "");

		// Proven: no unsolved opener's bound is below the best exact total.
		bool proven = best < INF;
		for (const Opener & o : openers)
			if (o.total < 0 && o.bound < best) proven = false;
		fprintf(stderr, "Exact search: %zu of %zu openers solved in %.1f s%s\n", solved, openers.size(),
			search_secs - gpu_secs, solved == openers.size() ? " (all)" : g_stop ? " (stopped)" : "");

		// Rank: solved by total (possible answers first on ties), then unsolved by bound.
		std::vector<size_t> rank(openers.size());
		for (size_t i = 0; i < rank.size(); i++) rank[i] = i;
		std::stable_sort(rank.begin(), rank.end(), [&](size_t a, size_t b) {
			const Opener & x = openers[a], & y = openers[b];
			const bool xs = x.total >= 0, ys = y.total >= 0;
			if (xs != ys) return xs;
			const int kx = xs ? x.total : x.bound, ky = ys ? y.total : y.bound;
			if (kx != ky) return kx < ky;
			if (x.st.candidate != y.st.candidate) return x.st.candidate > y.st.candidate;
			return eng.words()[x.guess] < eng.words()[y.guess];
		});

		// Games for the best k solved openers, built in parallel.
		size_t k = 0;
		while (k < rank.size() && openers[rank[k]].total >= 0 && openers[rank[k]].total < INF
			&& (top_k < 0 || (int)k < top_k))
			k++;
		std::vector<std::vector<Game1>> games(k);
		std::atomic<size_t> work{ 0 };
		std::string error;
		std::mutex error_mu;
		std::vector<std::thread> pool_threads;
		for (int t = 0; t < std::min<int>(threads, (int)k); t++)
			pool_threads.emplace_back([&] {
				for (size_t j; (j = work++) < k;)
				{
					try
					{
						std::vector<std::pair<int, uint8_t>> path;
						play_all(solver, eng, hard, cands, left, rule, openers[rank[j]].guess, path, games[j]);
					}
					catch (const std::exception & e)
					{
						std::lock_guard<std::mutex> lock(error_mu);
						error = e.what();
					}
				}
			});
		if (k) fprintf(stderr, "Writing the games of the best %zu openers...\n", k);
		for (auto & t : pool_threads) t.join();
		if (!error.empty()) throw std::runtime_error(error);

		// Guess-count distribution per opener with games (counting the guesses already played).
		auto spread = [&](const std::vector<Game1> & gs, int dist[MAX_GUESSES + 1], int & sum) {
			std::fill(dist, dist + MAX_GUESSES + 1, 0);
			sum = 0;
			for (const Game1 & g : gs)
			{
				const int c = played + (int)g.turns.size();
				dist[std::min(c, MAX_GUESSES)]++;
				sum += (int)g.turns.size();
			}
		};

		std::string header = "# wordle-gpu-solve: ";
		header += std::to_string(n) + " possible answers, " + std::to_string(openers.size()) + " openers, "
			+ (hard ? "hard" : "normal") + " mode";
		for (const Turn & t : game.turns()) header += ", after " + eng.words()[t.guess] + "=" + pattern_string(t.pat);
		char line[256];
		snprintf(line, sizeof line, "\n# GPU brute force %.2f s, exact search %.1f s: %zu of %zu openers solved%s\n",
			gpu_secs, search_secs - gpu_secs, solved, openers.size(), proven ? ", best proven optimal" : "");
		header += line;

		{
			std::ofstream f(prefix + "-openers.txt");
			f << header
				<< "# avg = expected guess number the game ends on; '>=' rows were not solved (lower bound).\n"
				<< "# total = guesses summed over all answers. * = could be the answer.\n"
				<< "# 1-6 = how many answers end on that guess (best " << k << " openers only).\n#\n";
			snprintf(line, sizeof line, "%6s  %-6s %9s %8s  %5s %5s %5s %5s %5s %5s  %5s\n",
				"rank", "guess", "avg", "total", "1", "2", "3", "4", "5", "6", "worst");
			f << line;
			for (size_t r = 0; r < rank.size(); r++)
			{
				const Opener & o = openers[rank[r]];
				const std::string w = eng.words()[o.guess] + (o.st.candidate ? "*" : "");
				if (o.total >= INF)
					snprintf(line, sizeof line, "%6zu  %-6s %9s %8s\n", r + 1, w.c_str(), "fails", "-");
				else if (o.total >= 0)
				{
					int len = snprintf(line, sizeof line, "%6zu  %-6s %9.4f %8d", r + 1, w.c_str(),
						played + (double)o.total / n, o.total);
					if (r < k)
					{
						int dist[MAX_GUESSES + 1], sum, worst = 0;
						spread(games[r], dist, sum);
						for (int c = 1; c <= MAX_GUESSES; c++)
						{
							len += snprintf(line + len, sizeof line - len, " %5d", dist[c]);
							if (dist[c]) worst = c;
						}
						snprintf(line + len, sizeof line - len, "  %5d\n", worst);
					}
					else
						snprintf(line + len, sizeof line - len, "\n");
				}
				else
					snprintf(line, sizeof line, "%6zu  %-6s >=%7.4f %8s\n", r + 1, w.c_str(),
						played + (double)o.bound / n, "");
				f << line;
			}
		}
		if (k)
		{
			std::ofstream f(prefix + "-games.txt");
			f << header << "# Every game of the optimal strategy for the best " << k
				<< " openers, shortest first.\n# answer  guesses  guess:colours ... (g green, y yellow, b gray)\n";
			for (size_t r = 0; r < k; r++)
			{
				const Opener & o = openers[rank[r]];
				auto & gs = games[r];
				std::sort(gs.begin(), gs.end(), [&](const Game1 & a, const Game1 & b) {
					if (a.turns.size() != b.turns.size()) return a.turns.size() < b.turns.size();
					return eng.words()[a.answer] < eng.words()[b.answer];
				});
				int dist[MAX_GUESSES + 1], sum;
				spread(gs, dist, sum);
				if (sum != o.total)
					fprintf(stderr, "warning: %s games add up to %d, solver said %d\n",
						eng.words()[o.guess].c_str(), sum, o.total);
				snprintf(line, sizeof line, "\n## %zu. %s  avg %.4f  total %d  ", r + 1, eng.words()[o.guess].c_str(),
					played + (double)o.total / n, o.total);
				f << line;
				for (int c = 1; c <= MAX_GUESSES; c++) f << c << ":" << dist[c] << (c < MAX_GUESSES ? " " : "\n");
				for (const Game1 & g : gs)
				{
					snprintf(line, sizeof line, "%s  %zu ", eng.words()[g.answer].c_str(), played + g.turns.size());
					f << line;
					for (auto [guess, pat] : g.turns) f << ' ' << eng.words()[guess] << ':' << pattern_string(pat);
					f << '\n';
				}
			}
		}

		// Summary like the files' first lines.
		if (best < INF)
		{
			printf("%s: %s avg %.4f, total %d over %d answers\n", proven ? "Best opener (proven)" : "Best opener so far",
				eng.words()[best_guess].c_str(), played + (double)best / n, best, n);
			if (k)
			{
				int dist[MAX_GUESSES + 1], sum, worst = 0;
				spread(games[0], dist, sum);
				for (int c = 1; c <= MAX_GUESSES; c++)
					if (dist[c]) { printf("  %d guesses: %5d\n", c, dist[c]); worst = c; }
				printf("  worst case: %d\n", worst);
			}
		}
		printf("Ranking of %zu openers (%zu solved): %s-openers.txt\n", openers.size(), solved, prefix.c_str());
		if (k) printf("Games of the best %zu: %s-games.txt\n", k, prefix.c_str());
	}
	catch (const std::exception & e)
	{
		fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
}
