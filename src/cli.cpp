// wordle-gpu-cli: rank next guesses from the command line.
#include "game.h"
#include "exact.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

// Positions with at most this many possible answers are solved exactly in interactive mode.
constexpr size_t AUTO_EXACT = 3000;

static void usage(const char * argv0)
{
	fprintf(stderr,
		"usage: %s [options] [guess=pattern ...]\n"
		"  pattern     5 of g/y/b (or 2/1/0), e.g. salet=bygbb\n"
		"  -T TIME     time budget per position, e.g. 30, 90s, 5m: brute force on the GPU,\n"
		"              then solve exactly until the best guess is proven or time runs out,\n"
		"              and show the best found (Ctrl-C also stops early)\n"
		"  -i          interactive: enter each guess and its colours, get the next suggestions\n"
		"  -H          hard mode\n"
		"  -a          any allowed word can be the answer (default: the original answers\n"
		"              plus every answer the NYT has used)\n"
		"  -o          only the original 2315 answers\n"
		"  -n N        show the best N guesses (default 15)\n"
		"  -L N        two-ply lookahead for only the best N by entropy (default all)\n"
		"  -1          skip the lookahead\n"
		"  -e N        exact expected guesses for the best N\n"
		"  -p          keep solving exactly until the best guess is proven optimal\n"
		"  -t N        exact solver threads (default all)\n"
		"  -C          don't read or write the opening cache (%s)\n"
		"  -w FILE     allowed-guess list (one word per line)\n"
		"  -A FILE     answer list\n"
		"  -d N        CUDA device (default 0)\n"
		"  -r          raw output: word buckets largest sum_sq entropy total by3 exact\n",
		argv0, default_cache_dir().c_str());
}

// "45", "45s", "2.5m", "1h" -> seconds; negative if malformed.
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

static std::atomic<bool> g_cancel{ false };
static void on_sigint(int) { g_cancel = true; }

struct Cli
{
	Engine & eng;
	Game & game;
	Game::Options opt;
	double budget = 0;       // seconds, 0 = none
	bool auto_exact = false; // prove small positions even without a budget
	int show = 15;
	bool raw = false;

	// Analyse the current position, stopping at the time budget or on Ctrl-C.
	void run()
	{
		Game::Options o = opt;
		if (auto_exact && game.candidates().size() <= AUTO_EXACT)
		{
			o.exact_top = std::max(o.exact_top, 10);
			o.exact_prove = true;
		}
		const auto t0 = std::chrono::steady_clock::now();
		auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
		double shown = -1;
		auto progress = [&](Game::Stage s, double f) {
			if (raw || !isatty(2) || secs() - shown < 0.2) return;
			shown = secs();
			fprintf(stderr, "\r%s %5.1f%%  %.1f s", s == Game::LOOKAHEAD ? "GPU brute force" : "exact search",
				f * 100, secs());
			if (budget > 0) fprintf(stderr, " of %.0f", budget);
			fprintf(stderr, "   ");
		};

		g_cancel = false;
		std::mutex mu;
		std::condition_variable cv;
		bool finished = false;
		std::thread timer;
		if (budget > 0)
			timer = std::thread([&] {
				std::unique_lock<std::mutex> lock(mu);
				if (!cv.wait_for(lock, std::chrono::duration<double>(budget), [&] { return finished; }))
					g_cancel = true;
			});
		auto old = signal(SIGINT, on_sigint);
		Analysis a;
		const bool ok = game.analyze(o, a, {}, progress, &g_cancel);
		signal(SIGINT, old);
		{
			std::lock_guard<std::mutex> lock(mu);
			finished = true;
		}
		cv.notify_all();
		if (timer.joinable()) timer.join();
		if (!raw && isatty(2)) fprintf(stderr, "\r%60s\r", "");
		print(a, ok, secs());
	}

	void print(const Analysis & a, bool complete, double secs)
	{
		const int n = (int)a.cands.size();
		if (raw)
		{
			for (const Suggestion & s : a.rows)
				printf("%s %d %d %d %.5f %d %d %d\n", eng.words()[s.guess].c_str(), s.st.buckets, s.st.largest,
					s.st.sum_sq, s.st.entropy, s.la.total, s.la.by3, s.exact);
			return;
		}
		if (!game.turns().empty() && game.turns().back().pat == ALL_GREEN)
		{
			printf("Solved in %zu!\n", game.turns().size());
			return;
		}
		printf("%d possible answer%s", n, n == 1 ? "" : "s");
		if (n > 0 && n <= 40)
		{
			printf(":");
			for (uint16_t c : a.cands) printf(" %s", eng.words()[eng.pool()[c]].c_str());
		}
		printf("\n");
		if (n == 0)
		{
			printf("No word fits these colours. Check them, or try -a (any valid word).\n");
			return;
		}
		if (a.rows.empty()) return;

		bool any_exact = false;
		for (int i = 0; i < show && i < (int)a.rows.size(); i++) any_exact |= a.rows[i].exact >= 0;
		printf("\n  #  guess   %-9s %-8s by%-2d  bits  groups  avg left  worst\n",
			a.have_lookahead ? "score>=" : "", any_exact ? "exact" : "", a.played + 3);
		for (int i = 0; i < show && i < (int)a.rows.size(); i++)
		{
			const Suggestion & s = a.rows[i];
			char lb[16] = "", ex[16] = "", by3[16] = "";
			if (a.have_lookahead && s.la.by3 >= 0)
			{
				snprintf(lb, sizeof lb, "%.4f", a.expected_lb(s));
				snprintf(by3, sizeof by3, "%5.1f%%", 100.0 * s.la.by3 / n);
			}
			if (s.exact >= INF) snprintf(ex, sizeof ex, "fails");
			else if (s.exact >= 0) snprintf(ex, sizeof ex, "%.4f", a.expected_exact(s));
			printf("%3d  %s%c  %-9s %-8s %6s  %4.2f  %6d  %8.2f  %5d\n", i + 1,
				eng.words()[s.guess].c_str(), s.st.candidate ? '*' : ' ', lb, ex, by3,
				s.st.entropy, s.st.buckets, (double)s.st.sum_sq / n, s.st.largest);
		}

		int solved = 0;
		for (const Suggestion & s : a.rows) solved += s.exact >= 0;
		printf("\n");
		if (a.proven)
			printf("%s is proven optimal", eng.words()[a.rows[0].guess].c_str());
		else if (!complete)
			printf("Stopped: best found so far, not proven");
		else
			printf("Done");
		printf(" (%.1f s", secs);
		if (!a.have_lookahead) printf(", GPU brute force not finished");
		if (solved) printf(", %d guess%s solved exactly", solved, solved == 1 ? "" : "es");
		if (a.cached) printf(", %d from the opening cache", a.cached);
		printf(").\n* = could be the answer. score = expected guess number the game ends on.\n");
	}

	// Adds a turn, or prints why it can't.
	bool add_turn(const std::string & word, const std::string & pat_text)
	{
		const int g = game.find(word);
		const auto pat = parse_pattern(pat_text);
		if (g < 0) { fprintf(stderr, "'%s' is not in the word list\n", word.c_str()); return false; }
		if (!pat) { fprintf(stderr, "bad pattern '%s' (5 of g/y/b, e.g. bygbb)\n", pat_text.c_str()); return false; }
		if (game.hard() && !game.rule().allows(word.c_str()))
		{
			fprintf(stderr, "hard mode: '%s' doesn't reuse every revealed letter\n", word.c_str());
			return false;
		}
		game.push(g, *pat);
		return true;
	}

	void interactive()
	{
		printf("Enter each guess and its colours, e.g. \"tarse bbygb\" (g green, y yellow, b gray).\n"
			"Commands: undo, new, quit. Ctrl-C stops a long search and shows the best so far.\n\n");
		run();
		for (std::string line;;)
		{
			printf("\nguess %zu> ", game.turns().size() + 1);
			fflush(stdout);
			if (!std::getline(std::cin, line)) { printf("\n"); return; }
			std::istringstream in(line);
			std::string a, b;
			in >> a >> b;
			for (char & c : a) c = (char)std::tolower((unsigned char)c);
			if (a.empty()) continue;
			if (a == "q" || a == "quit" || a == "exit") return;
			if (a == "u" || a == "undo") game.pop();
			else if (a == "n" || a == "new") game.clear();
			else
			{
				if (b.empty() && a.size() == 11 && a[5] == '=') { b = a.substr(6); a.resize(5); }
				if (!add_turn(a, b)) continue;
			}
			run();
			if (!game.turns().empty() && game.turns().back().pat == ALL_GREEN)
			{
				game.clear();
				printf("\nNew game.\n");
			}
		}
	}
};

int main(int argc, char ** argv)
{
	bool hard = false, raw = false, inter = false;
	Pool pool_kind = Pool::NYT;
	int show = 15, device = 0;
	double budget = 0;
	std::string allowed_path, answers_path;
	Game::Options opt;
	opt.cache_dir = default_cache_dir();
	for (int c; (c = getopt(argc, argv, "T:iHaon:L:1e:pt:Cw:A:d:rh")) != -1;)
	{
		switch (c)
		{
			case 'T':
				budget = parse_duration(optarg);
				if (budget <= 0) { fprintf(stderr, "bad time '%s' (e.g. 30, 90s, 5m)\n", optarg); return 2; }
				break;
			case 'i': inter = true; break;
			case 'H': hard = true; break;
			case 'a': pool_kind = Pool::ALL; break;
			case 'o': pool_kind = Pool::ORIGINAL; break;
			case 'n': show = atoi(optarg); break;
			case 'L': opt.lookahead_limit = atoi(optarg); break;
			case '1': opt.lookahead = false; break;
			case 'e': opt.exact_top = atoi(optarg); break;
			case 'p': opt.exact_prove = true; break;
			case 't': opt.threads = atoi(optarg); break;
			case 'C': opt.cache_dir.clear(); break;
			case 'w': allowed_path = optarg; break;
			case 'A': answers_path = optarg; break;
			case 'd': device = atoi(optarg); break;
			case 'r': raw = true; break;
			default: usage(argv[0]); return c == 'h' ? 0 : 2;
		}
	}
	// A budget is spent proving the best guess; it stops early once that is done.
	if (budget > 0) opt.exact_prove = true;

	try
	{
		WordLists lists = builtin_lists();
		if (!allowed_path.empty()) lists.allowed = read_list(allowed_path);
		if (!answers_path.empty()) lists.answers = read_list(answers_path);
		std::vector<int> pool = prepare_pool(lists, pool_kind);

		const auto t0 = std::chrono::steady_clock::now();
		Engine eng(lists.allowed, pool, device);
		if (!raw)
			fprintf(stderr, "%s: %d guesses x %d answers, tables in %.2f s\n", eng.device_name().c_str(),
				eng.num_guesses(), eng.pool_size(),
				std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

		Game game(eng, hard);
		Cli cli{ eng, game, opt, budget, inter && budget <= 0, show, raw };
		for (int i = optind; i < argc; i++)
		{
			const std::string arg = argv[i];
			if (arg.size() != 11 || arg[5] != '=')
			{
				fprintf(stderr, "bad turn '%s' (want word=pattern)\n", arg.c_str());
				return 2;
			}
			if (!cli.add_turn(arg.substr(0, 5), arg.substr(6))) return 2;
		}
		if (inter) cli.interactive();
		else cli.run();
	}
	catch (const std::exception & e)
	{
		fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
}
