#include "gamesdb.h"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>

std::string games_db_path()
{
	if (const char * x = getenv("XDG_DATA_HOME"); x && *x) return std::string(x) + "/wordle-gpu/games.sqlite";
	if (const char * h = getenv("HOME"); h && *h) return std::string(h) + "/.local/share/wordle-gpu/games.sqlite";
	return {};
}

std::string StoredGame::date() const
{
	if (puzzle < 0) return {};
	using namespace std::chrono;
	const year_month_day d{ sys_days{ year{ 2021 } / June / 19 } + days{ puzzle } };
	char buf[16];
	snprintf(buf, sizeof buf, "%04d-%02u-%02u", (int)d.year(), (unsigned)d.month(), (unsigned)d.day());
	return buf;
}

static std::vector<std::string> split(const std::string & s)
{
	std::vector<std::string> out;
	std::istringstream in(s);
	for (std::string w; in >> w;) out.push_back(w);
	return out;
}

GamesDb::GamesDb(const std::string & path, bool write) : path_(path)
{
	const int flags = write ? SQLITE_OPEN_READWRITE : SQLITE_OPEN_READONLY;
	if (sqlite3_open_v2(path.c_str(), &db_, flags, nullptr) != SQLITE_OK)
	{
		std::string err = db_ ? sqlite3_errmsg(db_) : "out of memory";
		sqlite3_close(db_);
		db_ = nullptr;
		throw std::runtime_error("cannot open " + path + ": " + err +
			" (fill it with wordle-gpu-import-games)");
	}
	sqlite3_busy_timeout(db_, 5000);
	try
	{
		if (query("SELECT 1 FROM sqlite_master WHERE name='games'").empty())
			throw std::runtime_error(path + " is not a wordle-gpu games database");
	}
	catch (...)
	{
		sqlite3_close(db_);
		throw;
	}
}

GamesDb::~GamesDb()
{
	sqlite3_close(db_);
}

GamesDb::Rows GamesDb::query(const std::string & sql, const std::vector<std::string> & args)
{
	sqlite3_stmt * st = nullptr;
	if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &st, nullptr) != SQLITE_OK)
		throw std::runtime_error(std::string("SQL error: ") + sqlite3_errmsg(db_));
	for (size_t i = 0; i < args.size(); i++)
		sqlite3_bind_text(st, (int)i + 1, args[i].c_str(), -1, SQLITE_TRANSIENT);
	Rows rows;
	int rc;
	while ((rc = sqlite3_step(st)) == SQLITE_ROW)
	{
		auto & row = rows.emplace_back();
		for (int c = 0; c < sqlite3_column_count(st); c++)
		{
			const auto * t = sqlite3_column_text(st, c);
			row.emplace_back(t ? (const char *)t : "");
		}
	}
	sqlite3_finalize(st);
	if (rc != SQLITE_DONE) throw std::runtime_error(std::string("SQL error: ") + sqlite3_errmsg(db_));
	return rows;
}

void GamesDb::exec(const std::string & sql)
{
	char * err = nullptr;
	if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK)
	{
		std::string msg = err ? err : "unknown";
		sqlite3_free(err);
		throw std::runtime_error("SQL error: " + msg);
	}
}

std::vector<GameSource> GamesDb::sources()
{
	std::vector<GameSource> out;
	const Rows rows = query(
		"SELECT s.id, s.name, ifnull(s.title,''), ifnull(s.url,''), ifnull(s.license,''), ifnull(s.kind,''), "
		"ifnull(s.notes,''), ifnull(s.imported,''), "
		"(SELECT count(*) FROM games g WHERE g.source=s.id), "
		"(SELECT count(*) FROM games g WHERE g.source=s.id AND g.guesses IS NOT NULL), "
		"(SELECT count(*) FROM daily d WHERE d.source=s.id) FROM sources s ORDER BY s.id");
	for (const auto & r : rows)
		out.push_back({ std::stoi(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7], std::stol(r[8]), std::stol(r[9]),
			std::stol(r[10]) });
	return out;
}

static const char * GAME_COLUMNS = "g.id, s.name, g.puzzle, g.answer, g.hard, g.player, g.variant, g.result, "
	"g.guesses, g.colors, g.consistent";

static StoredGame to_game(const std::vector<std::string> & r)
{
	auto num = [](const std::string & s, int none) { return s.empty() ? none : std::stoi(s); };
	StoredGame g;
	g.id = std::stol(r[0]);
	g.source = r[1];
	g.puzzle = num(r[2], -1);
	g.answer = r[3];
	g.hard = num(r[4], -1);
	g.player = r[5];
	g.variant = r[6];
	g.result = num(r[7], -1);
	g.guesses = split(r[8]);
	g.colors = split(r[9]);
	g.consistent = num(r[10], -1);
	return g;
}

std::optional<StoredGame> GamesDb::game(long id)
{
	const Rows r = query(std::string("SELECT ") + GAME_COLUMNS +
		" FROM games g JOIN sources s ON s.id=g.source WHERE g.id=?", { std::to_string(id) });
	if (r.empty()) return std::nullopt;
	return to_game(r[0]);
}

std::vector<StoredGame> GamesDb::games(const GameFilter & f)
{
	std::string where = " WHERE 1";
	std::vector<std::string> args;
	if (!f.source.empty())
	{
		where += " AND g.source=(SELECT id FROM sources WHERE name=?)";
		args.push_back(f.source);
	}
	if (!f.answer.empty())
	{
		where += " AND g.answer=?";
		args.push_back(f.answer);
	}
	if (f.puzzle >= 0)
	{
		where += " AND g.puzzle=?";
		args.push_back(std::to_string(f.puzzle));
	}
	if (f.words_only) where += " AND g.guesses IS NOT NULL";
	if (f.consistent) where += " AND g.consistent=1";
	const std::string from = std::string(" FROM games g JOIN sources s ON s.id=g.source") + where;

	std::vector<StoredGame> out;
	if (!f.random)
	{
		for (const auto & r : query(std::string("SELECT ") + GAME_COLUMNS + from + " ORDER BY g.id LIMIT " +
				std::to_string(f.limit) + " OFFSET " + std::to_string(f.offset), args))
			out.push_back(to_game(r));
		return out;
	}
	// Each source's games have consecutive ids, so a random id and the first match from
	// there gives a fair enough sample without sorting millions of rows.
	const Rows range = query("SELECT min(g.id), max(g.id)" + from, args);
	if (range.empty() || range[0][0].empty()) return out;
	const long lo = std::stol(range[0][0]), hi = std::stol(range[0][1]);
	std::mt19937_64 rng(std::random_device{}());
	std::uniform_int_distribution<long> pick(lo, hi);
	auto from_id = args;
	from_id.push_back("");
	const std::string sql = std::string("SELECT ") + GAME_COLUMNS + from + " AND g.id>=? ORDER BY g.id LIMIT 1";
	for (int i = 0; i < f.limit * 3 && (int)out.size() < f.limit; i++)
	{
		from_id.back() = std::to_string(pick(rng));
		const Rows r = query(sql, from_id);
		if (r.empty()) continue;
		StoredGame g = to_game(r[0]);
		bool dup = false;
		for (const auto & o : out) dup |= o.id == g.id;
		if (!dup) out.push_back(std::move(g));
	}
	return out;
}

std::vector<DayStats> GamesDb::day_stats(int puzzle)
{
	const std::string p = std::to_string(puzzle);
	std::vector<DayStats> out;
	auto num = [](const std::string & s) { return s.empty() ? -1.0 : atof(s.c_str()); };
	for (const auto & r : query("SELECT s.name, d.mode, d.games, d.hard_games, d.n1, d.n2, d.n3, d.n4, d.n5, d.n6, "
			"d.failed, d.percent, d.average FROM daily d JOIN sources s ON s.id=d.source WHERE d.puzzle=? "
			"ORDER BY s.id, d.mode LIKE '%-ps', d.mode DESC", { p }))
	{
		DayStats d;
		d.source = r[0];
		d.mode = r[1];
		d.games = atol(r[2].c_str());
		d.hard_games = r[3].empty() ? -1 : atol(r[3].c_str());
		for (int i = 0; i < 7; i++) d.n[i] = std::max(0.0, num(r[4 + i]));
		d.percent = r[11] == "1";
		d.average = num(r[12]);
		out.push_back(d);
	}
	for (const auto & r : query("SELECT s.name, count(*), sum(g.hard=1), sum(g.hard IS NULL), "
			"sum(g.result=1), sum(g.result=2), sum(g.result=3), sum(g.result=4), sum(g.result=5), sum(g.result=6), "
			"sum(g.result=0), avg(CASE WHEN g.result>0 THEN g.result END) "
			"FROM games g JOIN sources s ON s.id=g.source WHERE g.puzzle=? GROUP BY s.id ORDER BY s.id", { p }))
	{
		DayStats d;
		d.source = r[0];
		d.mode = "shared";
		d.games = atol(r[1].c_str());
		d.hard_games = atol(r[3].c_str()) == d.games ? -1 : atol(r[2].c_str());
		for (int i = 0; i < 7; i++) d.n[i] = atof(r[4 + i].c_str());
		d.average = num(r[11]);
		d.average_solved = true;
		out.push_back(d);
	}
	return out;
}

std::string GamesDb::answer(int puzzle)
{
	const std::string p = std::to_string(puzzle);
	const Rows r = query("SELECT answer FROM daily WHERE puzzle=? AND answer IS NOT NULL "
		"UNION ALL SELECT answer FROM games WHERE puzzle=? AND answer IS NOT NULL LIMIT 1", { p, p });
	return r.empty() ? std::string() : r[0][0];
}

std::pair<int, int> GamesDb::daily_puzzles()
{
	const Rows r = query("SELECT min(puzzle), max(puzzle) FROM daily");
	if (r.empty() || r[0][0].empty()) return { -1, -1 };
	return { std::stoi(r[0][0]), std::stoi(r[0][1]) };
}
