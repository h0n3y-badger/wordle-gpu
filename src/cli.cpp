// wordle-gpu-cli: rank next guesses from the command line.
#include "game.h"
#include "exact.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>

static void usage(const char * argv0)
{
	fprintf(stderr,
		"usage: %s [options] [guess=pattern ...]\n"
		"  pattern     5 of g/y/b (or 2/1/0), e.g. salet=bygbb\n"
		"  -H          hard mode\n"
		"  -a          any allowed word can be the answer (default: the 2315 answers)\n"
		"  -n N        show the best N guesses (default 15)\n"
		"  -L N        two-ply lookahead for only the best N by entropy (default all)\n"
		"  -1          skip the lookahead\n"
		"  -e N        exact expected guesses for the best N\n"
		"  -p          keep solving exactly until the best guess is proven optimal\n"
		"  -t N        exact solver threads (default all)\n"
		"  -w FILE     allowed-guess list (one word per line)\n"
		"  -A FILE     answer list\n"
		"  -d N        CUDA device (default 0)\n"
		"  -r          raw output: word buckets largest sum_sq entropy total by3 exact\n", argv0);
}

int main(int argc, char ** argv)
{
	bool hard = false, all_pool = false, raw = false;
	int show = 15, device = 0;
	std::string allowed_path, answers_path;
	Game::Options opt;
	for (int c; (c = getopt(argc, argv, "Han:L:1e:pt:w:A:d:rh")) != -1;)
	{
		switch (c)
		{
			case 'H': hard = true; break;
			case 'a': all_pool = true; break;
			case 'n': show = atoi(optarg); break;
			case 'L': opt.lookahead_limit = atoi(optarg); break;
			case '1': opt.lookahead = false; break;
			case 'e': opt.exact_top = atoi(optarg); break;
			case 'p': opt.exact_prove = true; break;
			case 't': opt.threads = atoi(optarg); break;
			case 'w': allowed_path = optarg; break;
			case 'A': answers_path = optarg; break;
			case 'd': device = atoi(optarg); break;
			case 'r': raw = true; break;
			default: usage(argv[0]); return c == 'h' ? 0 : 2;
		}
	}

	try
	{
		WordLists lists = builtin_lists();
		if (!allowed_path.empty()) lists.allowed = read_list(allowed_path);
		if (!answers_path.empty()) lists.answers = read_list(answers_path);
		std::vector<int> pool = prepare_pool(lists, all_pool);

		auto t0 = std::chrono::steady_clock::now();
		auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
		Engine eng(lists.allowed, pool, device);
		fprintf(stderr, "%s: %d guesses x %d answers, tables in %.2f s\n",
			eng.device_name().c_str(), eng.num_guesses(), eng.pool_size(), secs());

		Game game(eng, hard);
		for (int i = optind; i < argc; i++)
		{
			const std::string arg = argv[i];
			const size_t eq = arg.find('=');
			const int g = eq == 5 ? game.find(arg.substr(0, 5)) : -1;
			const auto pat = eq == 5 ? parse_pattern(arg.substr(6)) : std::nullopt;
			if (g < 0 || !pat)
			{
				fprintf(stderr, "bad turn '%s' (want word=pattern with an allowed word)\n", arg.c_str());
				return 2;
			}
			game.push(g, *pat);
		}

		double shown = -1;
		auto progress = [&](Game::Stage s, double f) {
			if (raw || (f < 1 && secs() - shown < 0.2)) return;
			shown = secs();
			fprintf(stderr, "\r%s %5.1f%%  %.1f s ", s == Game::LOOKAHEAD ? "lookahead" : "exact", f * 100, secs());
		};
		t0 = std::chrono::steady_clock::now();
		Analysis a;
		game.analyze(opt, a, {}, progress);
		fprintf(stderr, "\rdone in %.2f s%30s\n", secs(), "");

		const int n = (int)a.cands.size();
		if (raw)
		{
			for (const Suggestion & s : a.rows)
				printf("%s %d %d %d %.5f %d %d %d\n", eng.words()[s.guess].c_str(), s.st.buckets, s.st.largest,
					s.st.sum_sq, s.st.entropy, s.la.total, s.la.by3, s.exact);
			return 0;
		}
		printf("%d candidate%s left", n, n == 1 ? "" : "s");
		if (n > 0 && n <= 40)
		{
			printf(":");
			for (uint16_t c : a.cands) printf(" %s", eng.words()[eng.pool()[c]].c_str());
		}
		printf("\n");
		if (n == 0 || a.rows.empty()) return 0;

		printf("\n  #  guess  %-9s %-8s  by%-2d  bits  groups  avg left  worst\n",
			a.have_lookahead ? "score>=" : "", a.rows[0].exact >= 0 ? "exact" : "", a.played + 3);
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
			printf("%3d  %s%c %-9s %-8s %6s  %4.2f  %6d  %8.2f  %5d\n", i + 1,
				eng.words()[s.guess].c_str(), s.st.candidate ? '*' : ' ', lb, ex, by3,
				s.st.entropy, s.st.buckets, (double)s.st.sum_sq / n, s.st.largest);
		}
		if (a.proven) printf("\n%s is proven optimal.\n", eng.words()[a.rows[0].guess].c_str());
		printf("\n* = could be the answer. score = expected guess number the game ends on.\n");
	}
	catch (const std::exception & e)
	{
		fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
}
