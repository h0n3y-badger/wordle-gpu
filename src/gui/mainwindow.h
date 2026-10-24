#pragma once

#include "game.h"

#include <QMainWindow>

#include <atomic>
#include <memory>
#include <thread>

class Board;
class SuggestionModel;
class SuggestionFilter;
class QCheckBox;
class QComboBox;
class QLabel;
class QListWidget;
class QProgressBar;
class QPushButton;
class QTableView;
class QTabWidget;

class MainWindow : public QMainWindow
{
	Q_OBJECT
public:
	explicit MainWindow(WordLists lists, QWidget * parent = nullptr);
	~MainWindow() override;

private:
	Engine * engine();
	void stopWorker();
	void analyze(bool exact);
	void showAnalysis(std::shared_ptr<const Analysis> a, Game::Stage stage, double secs);
	void showProgress(Game::Stage stage, double f);
	void finished(bool ok, const QString & error);
	void showBreakdown();
	int selectedGuess() const;
	void useWord(const QString & w);
	QString validate(const QString & w) const;

	WordLists lists_;
	std::vector<int> pools_[2];  // answers, every allowed word
	std::unique_ptr<Engine> engines_[2];

	Board * board_;
	QCheckBox * hard_;
	QComboBox * pool_;
	QLabel * summary_;
	SuggestionModel * model_;
	SuggestionFilter * filter_;
	QTableView * table_;
	QPushButton * exact_btn_;
	QPushButton * stop_btn_;
	QTabWidget * tabs_;
	QListWidget * cands_;
	QListWidget * breakdown_;
	QLabel * breakdown_title_;
	QLabel * status_;
	QProgressBar * progress_;

	std::thread worker_;
	std::atomic<bool> cancel_{ false };
	int gen_ = 0;
	std::shared_ptr<const Analysis> shown_;
	Engine * shown_eng_ = nullptr; // the engine shown_ was computed with
	double look_secs_ = 0;
};
