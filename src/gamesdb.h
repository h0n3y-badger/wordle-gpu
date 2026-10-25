// The games database that wordle-gpu-import-games fills with published Wordle games
// and daily statistics: $XDG_DATA_HOME/wordle-gpu/games.sqlite.
#pragma once

#include <optional>
#include <string>
#include <vector>

struct sqlite3;

// $XDG_DATA_HOME/wordle-gpu/games.sqlite or ~/.local/share/wordle-gpu/games.sqlite.
std::string games_db_path();

struct GameSource
{
	int id;
	std::string name, title, url, license, kind, notes, imported;
	long games = 0;      // rows in games
	long with_words = 0; // of those with the guessed words
	long days = 0;       // rows in daily
};

struct StoredGame
{
	long id = 0;
	std::string source;               // source name
	int puzzle = -1;                  // NYT puzzle number, -1 if none
	std::string answer;               // empty if unknown
	int hard = -1;                    // 1 hard mode, 0 normal, -1 unknown
	std::string player, variant;
	int result = -1;                  // guesses used if solved, 0 failed, -1 unfinished
	std::vector<std::string> guesses; // empty when only the colours were shared
	std::vector<std::string> colors;  // "bygbg" per row
	int consistent = -1;              // 1/0, -1 unchecked

	std::string date() const; // the NYT puzzle's date, or empty
};

struct GameFilter
{
	std::string source;     // source name, empty = any
	std::string answer;     // empty = any
	int puzzle = -1;
	bool words_only = false; // only games with the guessed words
	bool consistent = false; // only games whose colours are right for the answer
	bool random = false;     // random order instead of by id
	int limit = 50;
	long offset = 0;
};

class GamesDb
{
public:
	// Throws if the file is missing or not a games database.
	explicit GamesDb(const std::string & path = games_db_path(), bool write = false);
	~GamesDb();
	GamesDb(const GamesDb &) = delete;
	GamesDb & operator=(const GamesDb &) = delete;

	const std::string & path() const { return path_; }
	std::vector<GameSource> sources();
	std::vector<StoredGame> games(const GameFilter & f);
	std::optional<StoredGame> game(long id);

	// Runs SQL with ? parameters bound to args, returning every row as text ("" for NULL).
	using Rows = std::vector<std::vector<std::string>>;
	Rows query(const std::string & sql, const std::vector<std::string> & args = {});
	void exec(const std::string & sql);

private:
	std::string path_;
	sqlite3 * db_ = nullptr;
};
