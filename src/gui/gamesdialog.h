// Browse the games database: games from every source (replay the ones with words), and
// each puzzle's numbers from WordleBot, the MCM data and shared results.
#pragma once

#include "gamesdb.h"

#include <QDialog>

#include <memory>
#include <string>
#include <vector>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QTabWidget;
class QTableWidget;
class QTextBrowser;

class GamesDialog : public QDialog
{
	Q_OBJECT
public:
	// allowed: every accepted guess, to show which words fit a colour-only row.
	// Throws if the database can't be opened.
	explicit GamesDialog(const std::vector<std::string> & allowed, QWidget * parent = nullptr);
	~GamesDialog() override;

	// The game to replay; id 0 if none.
	StoredGame selected() const;

private:
	void search(bool random);
	void showGame();
	void showPuzzle(int puzzle);

	std::unique_ptr<GamesDb> db_;
	const std::vector<std::string> & allowed_;
	std::vector<StoredGame> games_;

	QTabWidget * tabs_;
	QComboBox * source_;
	QLineEdit * answer_;
	QLineEdit * puzzle_filter_;
	QCheckBox * consistent_;
	QTableWidget * table_;
	QTextBrowser * details_;
	QSpinBox * puzzle_;
	QLabel * puzzle_title_;
	QTextBrowser * report_;
	QPushButton * open_;
};
