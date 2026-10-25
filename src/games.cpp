// wordle-gpu-games: browse the games database and score played games against the solver.
#include "game.h"
#include "exact.h"
#include "gamesdb.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <map>
#include <unistd.h>

static void usage(const char * argv0)
{
	fprintf(stderr,
		"usage: %s [options] [command] [args]\n"
		"commands:\n"
		"  sources           what has been imported (default)\n"
		"  stats [SOURCE..]  results per source: games, solved, average, guess spread\n"
		"  day DATE|PUZZLE   one NYT puzzle: WordleBot's and the MCM data's numbers, shared results\n"
		"  list              games, filtered by -s -w -p -W -c; -n N, -R random\n"
		"  show ID...        a game's rows and, with -x, every guess scored against the best\n"
		"  analyze           score every guess of games with words (-s, -n per source) and\n"
		"                    keep the results; then a summary per source\n"
		"options:\n"
		"  -D FILE     database (default %s)\n"
		"  -s SOURCE   only this source\n"
		"  -w WORD     only games with this answer\n"
		"  -p N        only NYT puzzle N\n"
		"  -W          only games with the guessed words\n"
		"  -c          only games whose colours fit the answer under the real rules\n"
		"  -n N        how many games (list: 20, analyze: 100 per source)\n"
		"  -R          random games instead of the first ones\n"
		"  -x          show: score each guess now (needs the GPU)\n"
		"  -q          score with the GPU's two-ply bounds only, no exact search (faster)\n"
		"  -a          any allowed word can be the answer (default: original + NYT answers)\n"
		"  -t N        exact solver threads (default all)\n"
		"  -d N        CUDA device (default 0)\n"
		"Scores are expected guess numbers the game ends on: 'lost' is how much worse the\n"
		"played guess is than the best one, playing perfectly afterwards. Fill the database\n"
		"with wordle-gpu-import-games; it stays on this machine.\n",
		argv0, games_db_path().c_str());
}

static std::atomic<bool> g_cancel{ false };
static void on_sigint(int) { g_cancel = true; }

static bool color_tty = false;

// Colour rows as tiles on a terminal, else as letters.
static std::string tiles(const std::string & word, const std::string & colors)
{
	if (!color_tty) return word.empty() ? colors : word + " " + colors;
	std::string out;
	for (int i = 0; i < 5; i++)
	{
		const char c = colors[i];
		out += c == 'g' ? "\x1b[1;97;48;2;83;141;78m" : c == 'y' ? "\x1b[1;97;48;2;181;159;59m" : "\x1b[1;97;48;2;58;58;60m";
		out += ' ';
		out += word.empty() ? ' ' : (char)toupper(word[i]);
		out += " \x1b[0m";
	}
	return out;
}

static std::string result_text(const StoredGame & g)
{
	return g.result > 0 ? std::to_string(g.result) + "/6" : g.result == 0 ? "X/6" : "unfinished";
}

static double num(const std::string & s) { return s.empty() ? NAN : atof(s.c_str()); }

struct Opts
{
	GameFilter filter;
	bool score_now = false, quick = false;
	Pool pool = Pool::NYT;
	int threads = 0, device = 0;
	bool limit_set = false;
};

// One played guess, scored at the position it was played in.
struct TurnEval
{
	int turn;
	std::string guess;
	int cands, cands_after;
	double expected_after; // average candidates the guess leaves
	double played, best;   // expected final score; played is NAN if it can't win in time
	std::string best_guess;
	bool exact;
};

static const char * pool_name(Pool p) { return p == Pool::ALL ? "all" : p == Pool::ORIGINAL ? "original" : "nyt"; }

class Scorer
{
public:
	Scorer(const Opts & o) : o_(o)
	{
		WordLists lists = builtin_lists();
		const std::vector<int> pool = prepare_pool(lists, o.pool);
		eng_ = std::make_unique<Engine>(lists.allowed, pool, o.device);
		fprintf(stderr, "%s: %d guesses x %d answers\n", eng_->device_name().c_str(), eng_->num_guesses(),
			eng_->pool_size());
	}

	// Scores each guess in turn; stops early where the colours fit no answer in the pool,
	// or on Ctrl-C (then returns false).
	bool score(const StoredGame & sg, std::vector<TurnEval> & out)
	{
		out.clear();
		const bool hard = sg.hard == 1;
		Game game(*eng_, hard);
		std::string key = hard ? "H" : "N";
		for (size_t t = 0; t < sg.guesses.size() && t < sg.colors.size(); t++)
		{
			const int g = game.find(sg.guesses[t]);
			const auto pat = parse_pattern(sg.colors[t]);
			if (g < 0 || !pat) return true;
			key += sg.guesses[t] + sg.colors[t];
			auto it = memo_.find(key);
			if (it == memo_.end())
			{
				TurnEval e;
				if (!evaluate(game, g, *pat, e)) return false;
				if (e.cands == 0) return true;
				it = memo_.emplace(key, e).first;
			}
			out.push_back(it->second);
			out.back().turn = (int)t + 1;
			game.push(g, *pat);
			if (*pat == ALL_GREEN) break;
		}
		return true;
	}

private:
	bool evaluate(Game & game, int g, uint8_t pat, TurnEval & e)
	{
		Game::Options opt;
		opt.exact_prove = !o_.quick;
		opt.threads = o_.threads;
		opt.cache_dir = default_cache_dir();
		Analysis a;
		if (!game.analyze(opt, a, {}, {}, &g_cancel)) return false;
		const int n = (int)a.cands.size();
		e.guess = eng_->words()[g];
		e.cands = n;
		if (n == 0 || a.rows.empty()) { e.cands = 0; return true; }
		const Suggestion * mine = nullptr;
		for (const Suggestion & s : a.rows)
			if (s.guess == g) mine = &s;
		if (!mine) { e.cands = 0; return true; } // hard mode didn't allow it
		e.cands_after = 0;
		for (uint16_t c : a.cands) e.cands_after += eng_->pat(g, c) == pat;
		e.expected_after = (double)mine->st.sum_sq / n;
		const Suggestion & best = a.rows[0];
		e.best_guess = eng_->words()[best.guess];
		e.exact = !o_.quick && best.exact >= 0;
		if (o_.quick)
		{
			e.best = a.expected_lb(best);
			e.played = a.expected_lb(*mine);
			return true;
		}
		int total = mine->exact;
		if (total < 0)
		{
			if (!solver_[game.hard()]) solver_[game.hard()] = std::make_unique<ExactSolver>(*eng_, game.hard(), o_.threads);
			// The two-ply bounds of each colour group let the search prune far more.
			const HardRule rule = game.rule();
			std::vector<Lookahead> la;
			std::vector<int> group_k, ex;
			if (!eng_->lookahead(a.cands, { g }, game.hard() ? &rule : nullptr, la, {}, &g_cancel, &group_k))
				return false;
			const std::vector<const int *> gk{ group_k.data() };
			if (!solver_[game.hard()]->totals(a.cands, { g }, MAX_GUESSES - a.played, rule, ex, {}, &g_cancel, nullptr, &gk))
				return false;
			total = ex[0];
			// Openers are the slow ones; keep them where the next analysis will find them.
			if (a.played == 0 && total > 0 && total < INF)
				if (const std::string path = opener_cache_path(*eng_, game.hard(), opt.cache_dir); !path.empty())
					std::ofstream(path, std::ios::app) << e.guess << ' ' << total << '\n';
		}
		e.best = best.exact >= 0 && best.exact < INF ? a.expected_exact(best) : NAN;
		e.played = total >= 0 && total < INF ? a.played + (double)total / n : NAN;
		return true;
	}

	const Opts & o_;
	std::unique_ptr<Engine> eng_;
	std::unique_ptr<ExactSolver> solver_[2];
	std::map<std::string, TurnEval> memo_; // by mode and the turns up to and including this guess
};

static void create_eval_table(GamesDb & db)
{
	db.exec("CREATE TABLE IF NOT EXISTS turn_eval("
		"game INTEGER NOT NULL, turn INTEGER NOT NULL, pool TEXT NOT NULL, "
		"guess TEXT, cands INTEGER, cands_after INTEGER, expected_after REAL, "
		"played REAL, best REAL, best_guess TEXT, exact INTEGER, "
		"PRIMARY KEY(game, turn, pool))");
}

static void store(GamesDb & db, long game, Pool pool, const std::vector<TurnEval> & ev)
{
	auto fmt = [](double v) { return std::isnan(v) ? std::string() : std::to_string(v); };
	for (const TurnEval & e : ev)
		db.query("INSERT OR REPLACE INTO turn_eval VALUES(?,?,?,?,?,?,?,nullif(?,''),nullif(?,''),?,?)",
			{ std::to_string(game), std::to_string(e.turn), pool_name(pool), e.guess, std::to_string(e.cands),
				std::to_string(e.cands_after), std::to_string(e.expected_after), fmt(e.played), fmt(e.best),
				e.best_guess, e.exact ? "1" : "0" });
}

static void print_game(const StoredGame & g, const std::vector<TurnEval> * ev)
{
	printf("game %ld  %s", g.id, g.source.c_str());
	if (g.puzzle >= 0) printf("  Wordle %d (%s)", g.puzzle, g.date().c_str());
	printf("  answer %s  %s%s", g.answer.empty() ? "unknown" : g.answer.c_str(), result_text(g).c_str(),
		g.hard == 1 ? "*" : "");
	if (!g.player.empty()) printf("  player %s", g.player.c_str());
	if (!g.variant.empty()) printf("  %s", g.variant.c_str());
	printf("\n");
	if (g.consistent == 0) printf("  (colours don't fit the answer under the real rules: the source's own feedback)\n");
	if (ev && !ev->empty())
		printf("  %-*s  answers left (expected)   score    best            lost\n", color_tty ? 20 : 11, "");
	double lost = 0;
	bool all = true;
	for (size_t i = 0; i < g.colors.size(); i++)
	{
		printf("  %s", tiles(i < g.guesses.size() ? g.guesses[i] : "", g.colors[i]).c_str());
		if (ev && i < ev->size())
		{
			const TurnEval & e = (*ev)[i];
			char played[16] = "fails", best[16] = "?";
			if (e.exact) snprintf(best, sizeof best, "fails");
			if (!std::isnan(e.played)) snprintf(played, sizeof played, "%.4f", e.played);
			if (!std::isnan(e.best)) snprintf(best, sizeof best, "%.4f", e.best);
			printf("  %5d -> %-5d (%7.1f)  %-7s  %s %-7s", e.cands, e.cands_after, e.expected_after, played,
				e.best_guess.c_str(), best);
			if (!std::isnan(e.played) && !std::isnan(e.best))
			{
				printf("  %+.4f%s", e.played - e.best, e.exact ? "" : " (bounds)");
				lost += e.played - e.best;
			}
			else all = false;
		}
		printf("\n");
	}
	if (ev && !ev->empty() && all) printf("  expected guesses lost to the best play: %.4f\n", lost);
}

static int cmd_sources(GamesDb & db)
{
	const auto src = db.sources();
	if (src.empty()) { printf("Nothing imported yet: run wordle-gpu-import-games.\n"); return 0; }
	printf("%-14s %-9s %10s %10s %6s  %s\n", "source", "kind", "games", "with words", "days", "licence");
	for (const auto & s : src)
		printf("%-14s %-9s %10ld %10ld %6ld  %s\n", s.name.c_str(), s.kind.c_str(), s.games, s.with_words, s.days,
			s.license.c_str());
	printf("\n");
	for (const auto & s : src)
		printf("%s: %s\n  %s\n  %s\n  imported %s\n", s.name.c_str(), s.title.c_str(), s.url.c_str(), s.notes.c_str(),
			s.imported.empty() ? "never" : s.imported.c_str());
	printf("\n%s (local only: don't redistribute it)\n", db.path().c_str());
	return 0;
}

static int cmd_stats(GamesDb & db, std::vector<std::string> names)
{
	if (names.empty())
		for (const auto & s : db.sources()) names.push_back(s.name);
	for (const auto & name : names)
	{
		const auto r = db.query(
			"SELECT count(*), count(DISTINCT player), sum(result>0), sum(result=0), sum(result IS NULL), "
			"avg(CASE WHEN result>0 THEN result END), sum(result=1), sum(result=2), sum(result=3), sum(result=4), "
			"sum(result=5), sum(result=6), sum(consistent=1), sum(consistent=0), sum(hard=1), sum(guesses IS NOT NULL), "
			"min(puzzle), max(puzzle) FROM games WHERE source=(SELECT id FROM sources WHERE name=?)", { name });
		const auto d = db.query("SELECT count(*), min(date), max(date), avg(average) FROM daily "
			"WHERE source=(SELECT id FROM sources WHERE name=?) AND mode IN ('normal','all')", { name });
		const long games = r.empty() ? 0 : atol(r[0][0].c_str());
		const long days = d.empty() ? 0 : atol(d[0][0].c_str());
		if (!games && !days) { printf("%s: no data\n\n", name.c_str()); continue; }
		printf("%s\n", name.c_str());
		if (games)
		{
			const auto & x = r[0];
			auto pct = [&](int i) { return 100.0 * atol(x[i].c_str()) / games; };
			printf("  %ld games", games);
			if (!x[1].empty() && x[1] != "0") printf(" by %s players", x[1].c_str());
			if (!x[16].empty()) printf(", Wordle %s to %s", x[16].c_str(), x[17].c_str());
			printf("\n  solved %.1f%%, failed %.1f%%", pct(2), pct(3));
			if (atol(x[4].c_str())) printf(", unfinished %.1f%%", pct(4));
			printf("; average %.3f guesses when solved\n  ", num(x[5]));
			for (int i = 1; i <= 6; i++) printf("%d: %.1f%%  ", i, pct(5 + i));
			printf("X: %.1f%%\n", pct(3));
			printf("  colours fit the answer: %.1f%% (%.1f%% don't", pct(12), pct(13));
			if (const long n = games - atol(x[12].c_str()) - atol(x[13].c_str())) printf(", %.1f%% unchecked", 100.0 * n / games);
			printf(")");
			if (atol(x[14].c_str())) printf("; hard mode %.1f%%", pct(14));
			if (atol(x[15].c_str()) && atol(x[15].c_str()) < games) printf("; with words %.1f%%", pct(15));
			printf("\n");
		}
		if (days)
			printf("  %ld days, %s to %s, average %.3f guesses\n", days, d[0][1].c_str(), d[0][2].c_str(), num(d[0][3]));
		printf("\n");
	}
	return 0;
}

static int cmd_day(GamesDb & db, const std::string & arg)
{
	std::string puzzle = arg;
	if (arg.find('-') != std::string::npos)
	{
		const auto r = db.query("SELECT CAST(julianday(?) - julianday('2021-06-19') AS INTEGER)", { arg });
		if (r.empty() || r[0][0].empty()) { fprintf(stderr, "bad date '%s' (YYYY-MM-DD)\n", arg.c_str()); return 2; }
		puzzle = r[0][0];
	}
	StoredGame probe;
	probe.puzzle = atoi(puzzle.c_str());
	const std::string answer = db.answer(probe.puzzle);
	printf("Wordle %d, %s: %s\n\n", probe.puzzle, probe.date().c_str(), answer.empty() ? "answer unknown" : answer.c_str());

	const auto stats = db.day_stats(probe.puzzle);
	for (const DayStats & d : stats)
	{
		printf("%-14s %-6s %8ld results", d.source.c_str(), d.mode.c_str(), d.games);
		if (d.hard_games >= 0) printf(" (%ld hard)", d.hard_games);
		if (d.average >= 0) printf("  average %.3f%s", d.average, d.average_solved ? " when solved" : "");
		double sum = 0;
		for (double x : d.n) sum += x;
		printf("\n%22s", "");
		for (int i = 0; i < 7; i++)
			printf("%s: %.1f%%  ", i == 6 ? "X" : std::to_string(i + 1).c_str(), sum > 0 ? 100 * d.n[i] / sum : 0.0);
		printf("\n");
	}
	if (stats.empty()) printf("No data for this puzzle.\n");
	return 0;
}

static int cmd_list(GamesDb & db, const Opts & o)
{
	for (const StoredGame & g : db.games(o.filter))
	{
		printf("%9ld  %-13s %5s  %-5s %-10s %s", g.id, g.source.c_str(), g.puzzle >= 0 ? std::to_string(g.puzzle).c_str() : "",
			g.answer.empty() ? "?" : g.answer.c_str(), (result_text(g) + (g.hard == 1 ? "*" : "")).c_str(),
			g.consistent == 0 ? "! " : "");
		for (size_t i = 0; i < g.colors.size(); i++)
			printf(" %s", i < g.guesses.size() ? (g.guesses[i] + ":" + g.colors[i]).c_str() : g.colors[i].c_str());
		printf("\n");
	}
	return 0;
}

static int cmd_show(GamesDb & db, const Opts & o, const std::vector<std::string> & ids)
{
	std::unique_ptr<Scorer> scorer;
	for (const auto & id : ids)
	{
		const auto g = db.game(atol(id.c_str()));
		if (!g) { fprintf(stderr, "no game %s\n", id.c_str()); continue; }
		std::vector<TurnEval> ev;
		if (!g->guesses.empty())
		{
			if (o.score_now)
			{
				if (!scorer)
				{
					create_eval_table(db);
					scorer = std::make_unique<Scorer>(o);
				}
				if (!scorer->score(*g, ev)) return 130;
				store(db, g->id, o.pool, ev);
			}
			else
				for (const auto & r : db.query("SELECT turn, guess, cands, cands_after, expected_after, played, best, "
						"best_guess, exact FROM turn_eval WHERE game=? AND pool=? ORDER BY turn", { id, pool_name(o.pool) }))
					ev.push_back({ atoi(r[0].c_str()), r[1], atoi(r[2].c_str()), atoi(r[3].c_str()), num(r[4]),
						num(r[5]), num(r[6]), r[7], r[8] == "1" });
		}
		print_game(*g, &ev);
		if (!g->guesses.empty() && ev.empty() && !o.score_now) printf("  (-x scores each guess)\n");
		printf("\n");
	}
	return 0;
}

static int cmd_analyze(GamesDb & db, Opts o)
{
	create_eval_table(db);
	std::vector<std::string> names;
	if (!o.filter.source.empty()) names.push_back(o.filter.source);
	else
		for (const auto & s : db.sources())
			if (s.with_words) names.push_back(s.name);
	if (!o.limit_set) o.filter.limit = 100;
	o.filter.words_only = true;
	o.filter.consistent = true; // other colours aren't positions the real game can reach

	Scorer scorer(o);
	auto old = signal(SIGINT, on_sigint);
	const auto t0 = std::chrono::steady_clock::now();
	int done = 0;
	for (const auto & name : names)
	{
		GameFilter f = o.filter;
		f.source = name;
		const auto games = db.games(f);
		int skipped = 0;
		for (size_t i = 0; i < games.size() && !g_cancel; i++)
		{
			if (!db.query("SELECT 1 FROM turn_eval WHERE game=? AND pool=? LIMIT 1",
					{ std::to_string(games[i].id), pool_name(o.pool) }).empty())
			{
				skipped++;
				continue;
			}
			std::vector<TurnEval> ev;
			if (!scorer.score(games[i], ev)) break;
			db.exec("BEGIN");
			store(db, games[i].id, o.pool, ev);
			db.exec("COMMIT");
			done++;
			if (isatty(2))
				fprintf(stderr, "\r%s: %zu/%zu games, %.0f s   ", name.c_str(), i + 1, games.size(),
					std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
		}
		if (isatty(2)) fprintf(stderr, "\r%60s\r", "");
		if (skipped) fprintf(stderr, "%s: %d games were already scored\n", name.c_str(), skipped);
	}
	signal(SIGINT, old);
	fprintf(stderr, "scored %d games in %.0f s%s\n", done,
		std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), g_cancel ? " (stopped)" : "");

	printf("\n%-14s %7s %7s  %-9s %-11s %-12s %s\n", "source", "games", "guesses", "optimal", "lost/guess",
		"lost/game", "luck: answers left vs expected");
	for (const auto & name : names)
	{
		const auto r = db.query(
			"SELECT count(DISTINCT e.game), count(*), avg(e.played - e.best < 1e-6), avg(e.played - e.best), "
			"sum(e.played - e.best) / count(DISTINCT e.game), avg(e.cands_after), avg(e.expected_after) "
			"FROM turn_eval e JOIN games g ON g.id=e.game WHERE g.source=(SELECT id FROM sources WHERE name=?) "
			"AND e.pool=? AND e.played IS NOT NULL AND e.best IS NOT NULL", { name, pool_name(o.pool) });
		if (r.empty() || r[0][0] == "0") continue;
		printf("%-14s %7s %7s  %6.1f%%   %-11.4f %-12.4f %.1f vs %.1f\n", name.c_str(), r[0][0].c_str(),
			r[0][1].c_str(), 100 * num(r[0][2]), num(r[0][3]), num(r[0][4]), num(r[0][5]), num(r[0][6]));
	}
	printf("\nlost = expected extra guesses compared with the best guess, playing perfectly after.\n");
	return g_cancel ? 130 : 0;
}

int main(int argc, char ** argv)
{
	Opts o;
	o.filter.limit = 20;
	std::string path = games_db_path();
	for (int c; (c = getopt(argc, argv, "D:s:w:p:Wcn:Rxqat:d:h")) != -1;)
	{
		switch (c)
		{
			case 'D': path = optarg; break;
			case 's': o.filter.source = optarg; break;
			case 'w': o.filter.answer = optarg; break;
			case 'p': o.filter.puzzle = atoi(optarg); break;
			case 'W': o.filter.words_only = true; break;
			case 'c': o.filter.consistent = true; break;
			case 'n': o.filter.limit = atoi(optarg); o.limit_set = true; break;
			case 'R': o.filter.random = true; break;
			case 'x': o.score_now = true; break;
			case 'q': o.quick = true; break;
			case 'a': o.pool = Pool::ALL; break;
			case 't': o.threads = atoi(optarg); break;
			case 'd': o.device = atoi(optarg); break;
			default: usage(argv[0]); return c == 'h' ? 0 : 2;
		}
	}
	for (char & ch : o.filter.answer) ch = (char)tolower((unsigned char)ch);
	color_tty = isatty(1);
	const std::string cmd = optind < argc ? argv[optind++] : "sources";
	std::vector<std::string> args(argv + optind, argv + argc);
	try
	{
		GamesDb db(path, cmd == "analyze" || (cmd == "show" && o.score_now));
		if (cmd == "sources") return cmd_sources(db);
		if (cmd == "stats") return cmd_stats(db, args);
		if (cmd == "day" && args.size() == 1) return cmd_day(db, args[0]);
		if (cmd == "list") return cmd_list(db, o);
		if (cmd == "show" && !args.empty()) return cmd_show(db, o, args);
		if (cmd == "analyze") return cmd_analyze(db, o);
		usage(argv[0]);
		return 2;
	}
	catch (const std::exception & e)
	{
		fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
}
