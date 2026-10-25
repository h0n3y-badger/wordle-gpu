#include "mainwindow.h"
#include "board.h"
#include "colors.h"
#include "model.h"
#include "exact.h"
#ifdef WORDLE_GPU_GAMES
#include "gamesdialog.h"
#endif

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QElapsedTimer>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableView>
#include <QVBoxLayout>

// Positions with at most this many possible answers are solved exactly without asking.
constexpr int AUTO_EXACT = 3000;

MainWindow::MainWindow(WordLists lists, QWidget * parent) : QMainWindow(parent), lists_(std::move(lists))
{
	for (Pool p : { Pool::NYT, Pool::ORIGINAL, Pool::ALL }) pools_[(int)p] = prepare_pool(lists_, p);
	setWindowTitle(tr("Wordle GPU Solver"));

	// Left: board and options.
	auto * left = new QWidget;
	auto * lv = new QVBoxLayout(left);
	board_ = new Board;
	board_->setValidator([this](const QString & w) { return validate(w); });
	lv->addWidget(board_, 0, Qt::AlignHCenter);
	auto * buttons = new QHBoxLayout;
	auto * undo = new QPushButton(tr("Undo row"));
	auto * reset = new QPushButton(tr("New game"));
	buttons->addWidget(undo);
	buttons->addWidget(reset);
#ifdef WORDLE_GPU_GAMES
	auto * saved = new QPushButton(tr("Saved games…"));
	saved->setToolTip(tr("Replay a game from the games database (fill it with wordle-gpu-import-games)."));
	buttons->addWidget(saved);
	connect(saved, &QPushButton::clicked, this, &MainWindow::openGame);
#endif
	lv->addLayout(buttons);

	// Replaying a saved game: where we are, and buttons to step through it (above the suggestions).
	replay_bar_ = new QWidget;
	auto * rv = new QHBoxLayout(replay_bar_);
	rv->setContentsMargins(0, 0, 0, 6);
	replay_info_ = new QLabel;
	replay_info_->setWordWrap(true);
	replay_info_->setTextFormat(Qt::RichText);
	auto * rb = new QVBoxLayout;
	replay_prev_ = new QPushButton(tr("◀ Previous guess"));
	replay_next_ = new QPushButton(tr("Next guess ▶"));
	auto * replay_close = new QPushButton(tr("Close game"));
	rb->addWidget(replay_prev_);
	rb->addWidget(replay_next_);
	rb->addWidget(replay_close);
	rv->addWidget(replay_info_, 1);
	rv->addLayout(rb);
	replay_bar_->hide();
	connect(replay_prev_, &QPushButton::clicked, this, [this] { showReplayTurn(board_->turns().size() - 1); });
	connect(replay_next_, &QPushButton::clicked, this, [this] { showReplayTurn(board_->turns().size() + 1); });
	connect(replay_close, &QPushButton::clicked, this, &MainWindow::endReplay);
	replay_prev_->setObjectName("replay_prev");
	replay_next_->setObjectName("replay_next");
	hard_ = new QCheckBox(tr("Hard mode"));
	lv->addWidget(hard_);
	pool_ = new QComboBox;
	pool_->addItem(tr("Answers: original + NYT picks (%1)").arg(pools_[(int)Pool::NYT].size()));
	pool_->addItem(tr("Answers: original list (%1)").arg(pools_[(int)Pool::ORIGINAL].size()));
	pool_->addItem(tr("Answers: any valid word (%1)").arg(pools_[(int)Pool::ALL].size()));
	pool_->setToolTip(tr("Words that may be the answer. The NYT has picked %1 answers outside the "
		"original list so far;\nrun wordle-gpu-update-answers now and then to fetch new ones.")
		.arg(pools_[(int)Pool::NYT].size() - pools_[(int)Pool::ORIGINAL].size()));
	lv->addWidget(pool_);
	auto * help = new QLabel(tr("Type a guess, click its tiles to match the colours you got "
		"(right-click goes back, or keys 1-5), then press Enter.\n"
		"Double-click a suggestion or a possible answer to type it in."));
	help->setWordWrap(true);
	help->setEnabled(false);
	lv->addWidget(help);
	lv->addStretch();

	// Right: suggestions on top, possible answers and the selected guess's breakdown below.
	auto * top = new QWidget;
	auto * tv = new QVBoxLayout(top);
	tv->setContentsMargins(0, 0, 0, 0);
	summary_ = new QLabel;
	summary_->setWordWrap(true);
	QFont big = summary_->font();
	big.setPointSizeF(big.pointSizeF() * 1.25);
	summary_->setFont(big);
	summary_->setTextFormat(Qt::RichText);
	tv->addWidget(replay_bar_);
	tv->addWidget(summary_);
	auto * filters = new QHBoxLayout;
	auto * search = new QLineEdit;
	search->setPlaceholderText(tr("Find a word…"));
	search->setClearButtonEnabled(true);
	auto * only = new QCheckBox(tr("Only possible answers"));
	exact_btn_ = new QPushButton(tr("Solve exactly"));
	exact_btn_->setToolTip(tr("Exact search on the CPU until the best guess is proven optimal.\n"
		"Seconds once a guess is in; the opening move can take a long time."));
	stop_btn_ = new QPushButton(tr("Stop"));
	stop_btn_->setEnabled(false);
	filters->addWidget(search, 1);
	filters->addWidget(only);
	filters->addWidget(exact_btn_);
	filters->addWidget(stop_btn_);
	tv->addLayout(filters);

	model_ = new SuggestionModel(this);
	filter_ = new SuggestionFilter(this);
	filter_->setSourceModel(model_);
	filter_->setSortRole(SuggestionModel::SortRole);
	table_ = new QTableView;
	table_->setModel(filter_);
	table_->setSortingEnabled(true);
	table_->sortByColumn(SuggestionModel::RANK, Qt::AscendingOrder);
	table_->setSelectionBehavior(QAbstractItemView::SelectRows);
	table_->setSelectionMode(QAbstractItemView::SingleSelection);
	table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
	table_->verticalHeader()->hide();
	table_->verticalHeader()->setDefaultSectionSize(table_->fontMetrics().height() + 6);
	table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	table_->horizontalHeader()->setSectionResizeMode(SuggestionModel::RANK, QHeaderView::ResizeToContents);
	tv->addWidget(table_);

	tabs_ = new QTabWidget;
	cands_ = new QListWidget;
	cands_->setFlow(QListView::LeftToRight);
	cands_->setWrapping(true);
	cands_->setResizeMode(QListView::Adjust);
	cands_->setUniformItemSizes(true);
	cands_->setSpacing(2);
	auto * bd = new QWidget;
	auto * bv = new QVBoxLayout(bd);
	breakdown_title_ = new QLabel;
	breakdown_ = new QListWidget;
	breakdown_->setIconSize(colors::pattern_icon(0, 14).size());
	breakdown_->setWordWrap(true);
	bv->addWidget(breakdown_title_);
	bv->addWidget(breakdown_);
	tabs_->addTab(cands_, tr("Possible answers"));
	tabs_->addTab(bd, tr("How the selected guess splits them"));

	auto * right = new QSplitter(Qt::Vertical);
	right->addWidget(top);
	right->addWidget(tabs_);
	right->setStretchFactor(0, 3);
	right->setStretchFactor(1, 1);

	auto * split = new QSplitter;
	split->addWidget(left);
	split->addWidget(right);
	split->setStretchFactor(1, 1);
	setCentralWidget(split);

	status_ = new QLabel;
	progress_ = new QProgressBar;
	progress_->setRange(0, 1000);
	progress_->setMaximumWidth(220);
	progress_->hide();
	statusBar()->addWidget(status_, 1);
	statusBar()->addPermanentWidget(progress_);
	auto * gpu = new QLabel(QString::fromStdString(engine()->device_name()));
	gpu->setEnabled(false);
	statusBar()->addPermanentWidget(gpu);

	connect(board_, &Board::turnsChanged, this, [this] { analyze(false); });
	connect(board_, &Board::rejected, this, [this](const QString & why) { statusBar()->showMessage(why, 4000); });
	connect(undo, &QPushButton::clicked, this, [this] { board_->undo(); board_->setFocus(); });
	connect(reset, &QPushButton::clicked, this, [this] { board_->reset(); board_->setFocus(); });
	connect(hard_, &QCheckBox::toggled, this, [this] { analyze(false); board_->setFocus(); });
	connect(pool_, &QComboBox::currentIndexChanged, this, [this] { analyze(false); board_->setFocus(); });
	connect(search, &QLineEdit::textChanged, filter_, &SuggestionFilter::setPrefix);
	connect(only, &QCheckBox::toggled, filter_, &SuggestionFilter::setOnlyCandidates);
	connect(exact_btn_, &QPushButton::clicked, this, [this] { analyze(true); });
	connect(stop_btn_, &QPushButton::clicked, this, [this] { cancel_ = true; });
	connect(table_->selectionModel(), &QItemSelectionModel::currentRowChanged, this, &MainWindow::showBreakdown);
	connect(table_, &QTableView::doubleClicked, this, [this](const QModelIndex & idx) {
		useWord(QString::fromStdString(engine()->words()[idx.data(SuggestionModel::GuessRole).toInt()]));
	});
	connect(cands_, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem * it) { useWord(it->text()); });

	board_->setObjectName("board");
	exact_btn_->setObjectName("exact");
	stop_btn_->setObjectName("stop");
	table_->setObjectName("table");
	tabs_->setObjectName("tabs");
	hard_->setObjectName("hard");
	pool_->setObjectName("pool");

	resize(1280, 800);
	board_->setFocus();
	analyze(false);
}

MainWindow::~MainWindow()
{
	stopWorker();
}

Engine * MainWindow::engine()
{
	const int i = pool_ ? pool_->currentIndex() : 0;
	if (!engines_[i])
	{
		QApplication::setOverrideCursor(Qt::WaitCursor);
		try
		{
			engines_[i] = std::make_unique<Engine>(lists_.allowed, pools_[i]);
		}
		catch (...)
		{
			QApplication::restoreOverrideCursor();
			throw;
		}
		QApplication::restoreOverrideCursor();
	}
	return engines_[i].get();
}

QString MainWindow::validate(const QString & w) const
{
	const std::string s = w.toStdString();
	if (!std::binary_search(lists_.allowed.begin(), lists_.allowed.end(), s))
		return tr("\"%1\" is not in the word list.").arg(w.toUpper());
	if (hard_->isChecked())
	{
		HardRule r;
		for (const Board::Row & t : board_->turns()) r.add(t.word.toStdString().data(), t.pattern());
		if (!r.allows(s.data())) return tr("Hard mode: \"%1\" doesn't reuse every revealed letter.").arg(w.toUpper());
	}
	return {};
}

void MainWindow::openGame()
{
#ifdef WORDLE_GPU_GAMES
	try
	{
		GamesDialog d(lists_.allowed, this);
		if (d.exec() != QDialog::Accepted) return;
		const StoredGame g = d.selected();
		if (g.id) replayGame(g.id);
	}
	catch (const std::exception & e)
	{
		QMessageBox::warning(this, windowTitle(), QString::fromStdString(e.what()));
	}
#endif
}

void MainWindow::replayGame(long id)
{
#ifdef WORDLE_GPU_GAMES
	std::optional<StoredGame> g;
	try
	{
		g = GamesDb().game(id);
	}
	catch (const std::exception & e)
	{
		QMessageBox::warning(this, windowTitle(), QString::fromStdString(e.what()));
		return;
	}
	if (!g || g->guesses.empty())
	{
		statusBar()->showMessage(tr("Game %1 isn't in the database, or has no words to replay.").arg(id), 4000);
		return;
	}
	replay_words_.clear();
	replay_colors_.clear();
	for (size_t i = 0; i < g->guesses.size() && i < g->colors.size(); i++)
	{
		replay_words_ << QString::fromStdString(g->guesses[i]);
		replay_colors_ << QString::fromStdString(g->colors[i]);
	}
	replay_title_ = tr("Game %1 from %2").arg(g->id).arg(QString::fromStdString(g->source));
	if (g->puzzle >= 0) replay_title_ += tr(", Wordle %1").arg(g->puzzle);
	if (!g->player.empty()) replay_title_ += QStringLiteral(", ") + QString::fromStdString(g->player);
	replay_result_ = (g->answer.empty() ? tr("answer unknown") : tr("answer %1").arg(QString::fromStdString(g->answer).toUpper()))
		+ ", " + (g->result > 0 ? tr("solved in %1").arg(g->result) : g->result == 0 ? tr("failed") : tr("unfinished"));
	if (g->consistent == 0) replay_result_ += tr("; its colours aren't the real game's");

	// The game's mode, and an answer list that has its answer.
	const QSignalBlocker b1(hard_), b2(pool_);
	hard_->setChecked(g->hard == 1);
	if (!g->answer.empty())
	{
		auto it = std::lower_bound(lists_.allowed.begin(), lists_.allowed.end(), g->answer);
		const int w = it != lists_.allowed.end() && *it == g->answer ? (int)(it - lists_.allowed.begin()) : -1;
		const auto & nyt = pools_[(int)Pool::NYT];
		if (pool_->currentIndex() == (int)Pool::ORIGINAL || !std::binary_search(nyt.begin(), nyt.end(), w))
			pool_->setCurrentIndex(std::binary_search(nyt.begin(), nyt.end(), w) ? (int)Pool::NYT : (int)Pool::ALL);
	}
	replaying_ = true;
	replay_bar_->show();
	showReplayTurn(0);
#else
	Q_UNUSED(id);
#endif
}

void MainWindow::showReplayTurn(int turn)
{
	if (!replaying_) return;
	turn = std::clamp(turn, 0, (int)replay_words_.size());
	QList<Board::Row> rows;
	for (int i = 0; i < turn; i++)
	{
		Board::Row r;
		r.word = replay_words_[i];
		const auto pat = parse_pattern(replay_colors_[i].toStdString());
		for (int k = 0; k < 5; k++) r.digit[k] = pat ? pat_digit(*pat, k) : 0;
		rows << r;
	}
	board_->setTurns(rows); // analyses the new position
	if (turn < replay_words_.size()) board_->setEditWord(replay_words_[turn]);
	replay_prev_->setEnabled(turn > 0);
	replay_next_->setEnabled(turn < replay_words_.size());
	updateReplayInfo();
	board_->setFocus();
}

void MainWindow::endReplay()
{
	replaying_ = false;
	replay_bar_->hide();
	board_->setEditWord(QString());
}

// Where the replay is, and how the guess played next ranks among the suggestions.
void MainWindow::updateReplayInfo()
{
	if (!replaying_) return;
	const QList<Board::Row> turns = board_->turns();
	bool same = turns.size() <= replay_words_.size();
	for (int i = 0; same && i < turns.size(); i++)
		same = turns[i].word == replay_words_[i] && pattern_string((uint8_t)turns[i].pattern()) == replay_colors_[i].toStdString();
	if (!same)
	{
		endReplay(); // the board was edited
		return;
	}
	const int k = (int)turns.size();
	QString text = QStringLiteral("<b>%1</b><br>%2.<br>").arg(replay_title_.toHtmlEscaped(), replay_result_.toHtmlEscaped());
	if (k >= replay_words_.size())
	{
		replay_info_->setText(text + tr("End of the game."));
		return;
	}
	const QString played = replay_words_[k].toUpper();
	text += tr("Guess %1: <b>%2</b>").arg(k + 1).arg(played);
	if (shown_ && shown_eng_ && !shown_->rows.empty() && (int)shown_->played == k)
	{
		const auto & a = *shown_;
		auto score = [&](const Suggestion & s) {
			if (s.exact >= INF) return tr("can't win in time");
			if (s.exact >= 0) return tr("average score %1").arg(a.expected_exact(s), 0, 'f', 4);
			return a.have_lookahead ? tr("average score ≥ %1").arg(a.expected_lb(s), 0, 'f', 4) : QString();
		};
		const std::string w = replay_words_[k].toStdString();
		int rank = -1;
		for (size_t i = 0; i < a.rows.size(); i++)
			if (shown_eng_->words()[a.rows[i].guess] == w) rank = (int)i;
		if (rank < 0)
			text += tr(" (not allowed here in hard mode)");
		else
		{
			const Suggestion & best = a.rows[0];
			text += tr(", ranked %1 of %2, %3").arg(rank + 1).arg(a.rows.size()).arg(score(a.rows[rank]));
			if (best.exact >= INF)
				text += tr(" (no guess can)");
			else if (rank > 0)
				text += tr("; best %1, %2").arg(QString::fromStdString(shown_eng_->words()[best.guess]).toUpper()).arg(score(best));
			if (rank > 0 && a.rows[rank].exact >= 0 && a.rows[rank].exact < INF && best.exact >= 0)
				text += tr(" (%1 guesses lost)").arg(a.expected_exact(a.rows[rank]) - a.expected_exact(best), 0, 'f', 4);
		}
	}
	replay_info_->setText(text + '.');
}

void MainWindow::useWord(const QString & w)
{
	board_->setEditWord(w.left(5));
	board_->setFocus();
}

void MainWindow::stopWorker()
{
	cancel_ = true;
	if (worker_.joinable()) worker_.join();
	cancel_ = false;
}

void MainWindow::analyze(bool exact)
{
	stopWorker();
	const int gen = ++gen_;
	Engine * eng;
	try { eng = engine(); }
	catch (const std::exception & e) { finished(false, e.what()); return; }

	auto game = std::make_shared<Game>(*eng, hard_->isChecked());
	for (const Board::Row & t : board_->turns()) game->push(game->find(t.word.toStdString()), (uint8_t)t.pattern());

	Game::Options opt;
	const size_t n = game->candidates().size();
	if (exact || n <= AUTO_EXACT)
	{
		opt.exact_top = 10;
		opt.exact_prove = true;
	}
	opt.threads = std::max(1, (int)std::thread::hardware_concurrency() - 1);
	opt.cache_dir = default_cache_dir();

	exact_btn_->setEnabled(false);
	stop_btn_->setEnabled(true);
	progress_->setValue(0);
	progress_->show();
	status_->setText(tr("Scoring every guess on the GPU…"));

	worker_ = std::thread([this, gen, game, opt] {
		QElapsedTimer timer;
		timer.start();
		QElapsedTimer since_progress;
		since_progress.start();
		auto report = [&](Game::Stage stage, const Analysis & a) {
			auto copy = std::make_shared<const Analysis>(a);
			const double secs = timer.elapsed() / 1000.0;
			QMetaObject::invokeMethod(this, [this, gen, copy, stage, secs] {
				if (gen == gen_) showAnalysis(copy, stage, secs);
			}, Qt::QueuedConnection);
		};
		auto progress = [&](Game::Stage stage, double f) {
			if (since_progress.elapsed() < 50 && f < 1) return;
			since_progress.restart();
			QMetaObject::invokeMethod(this, [this, gen, stage, f] {
				if (gen == gen_) showProgress(stage, f);
			}, Qt::QueuedConnection);
		};
		bool ok = false;
		QString error;
		try
		{
			Analysis a;
			ok = game->analyze(opt, a, report, progress, &cancel_);
		}
		catch (const std::exception & e) { error = e.what(); }
		QMetaObject::invokeMethod(this, [this, gen, ok, error] {
			if (gen == gen_) finished(ok, error);
		}, Qt::QueuedConnection);
	});
}

void MainWindow::showProgress(Game::Stage stage, double f)
{
	progress_->setValue((int)(f * 1000));
	if (stage == Game::LOOKAHEAD)
		status_->setText(tr("GPU: trying every first guess × every colour result × every second guess…"));
	else if (stage == Game::EXACT)
		status_->setText(tr("Exact search on the CPU…"));
}

void MainWindow::finished(bool ok, const QString & error)
{
	progress_->hide();
	stop_btn_->setEnabled(false);
	const bool proven = shown_ && shown_->proven;
	exact_btn_->setEnabled(!proven && shown_ && !shown_->rows.empty());
	if (!error.isEmpty())
	{
		status_->setText(tr("Error: %1").arg(error));
		QMessageBox::critical(this, windowTitle(), error);
	}
	else if (!ok)
		status_->setText(tr("Stopped."));
}

int MainWindow::selectedGuess() const
{
	const QModelIndex cur = table_->currentIndex();
	return cur.isValid() ? cur.data(SuggestionModel::GuessRole).toInt() : -1;
}

void MainWindow::showAnalysis(std::shared_ptr<const Analysis> a, Game::Stage stage, double secs)
{
	Engine * eng = engine();
	// Follow the best guess unless another row was picked.
	const int keep = table_->currentIndex().row() > 0 ? selectedGuess() : -1;
	const bool fresh = !shown_ || shown_->cands != a->cands || stage == Game::STATS;
	shown_ = a;
	shown_eng_ = eng;
	model_->set(a, eng);

	// Keep the selection on the same word, or pick the best one.
	int row = 0;
	if (keep >= 0 && !fresh)
		for (int i = 0; i < filter_->rowCount(); i++)
			if (filter_->index(i, 0).data(SuggestionModel::GuessRole).toInt() == keep) { row = i; break; }
	if (filter_->rowCount() > 0)
		table_->setCurrentIndex(filter_->index(row, SuggestionModel::GUESS));
	if (fresh) table_->scrollToTop();

	const int n = (int)a->cands.size();
	auto word = [&](int g) { return QString::fromStdString(eng->words()[g]).toUpper(); };
	if (stage == Game::STATS)
	{
		cands_->clear();
		for (uint16_t c : a->cands) cands_->addItem(QString::fromStdString(eng->words()[eng->pool()[c]]));
		tabs_->setTabText(0, tr("Possible answers (%1)").arg(n));
	}

	QString text;
	if (board_->solved())
		text = tr("Solved in %1!").arg(a->played);
	else if (n == 0)
		text = tr("<b>No word fits these colours.</b> Check the tiles, or switch the answer list.");
	else if (a->played >= MAX_GUESSES)
		text = tr("Out of guesses. %n word(s) were still possible.", "", n);
	else if (n == 1)
		text = tr("The answer is <b>%1</b>.").arg(word(eng->pool()[a->cands[0]]));
	else if (!a->rows.empty())
	{
		const Suggestion & best = a->rows[0];
		text = tr("%1 possible answers. Best next guess: <b>%2</b>").arg(n).arg(word(best.guess));
		if (best.exact >= INF)
			text = tr("%1 possible answers, and no guess can win in time.").arg(n);
		else if (best.exact >= 0)
			text += tr(" — average score %1").arg(a->expected_exact(best), 0, 'f', 4);
		else if (best.la.by3 >= 0)
			text += tr(" — average score ≥ %1").arg(a->expected_lb(best), 0, 'f', 4);
		if (a->proven) text += tr(" <span style='color:%1'>★ proven optimal</span>").arg(colors::GREEN.name());
	}
	summary_->setText(text);
	updateReplayInfo();

	if (stage == Game::LOOKAHEAD) look_secs_ = secs;
	if (a->rows.empty())
		status_->clear();
	else if (stage == Game::STATS)
		status_->setText(tr("Scored %1 guesses on the GPU.").arg(a->rows.size()));
	else if (stage == Game::LOOKAHEAD)
		status_->setText(tr("Brute-forced %1 first × %2 second guesses × %3 answers in %4 s.")
			.arg(a->rows.size()).arg(eng->num_guesses()).arg(n).arg(secs, 0, 'f', 2));
	else
	{
		int solved = 0;
		for (const Suggestion & s : a->rows) solved += s.exact >= 0;
		QString t = tr("GPU brute force %1 s, then %2 guesses solved exactly")
			.arg(look_secs_, 0, 'f', 2).arg(solved);
		if (a->cached) t += tr(" (%1 from the opening cache)").arg(a->cached);
		if (solved > a->cached) t += tr(" in %1 s").arg(secs - look_secs_, 0, 'f', 2);
		status_->setText(t + '.');
	}
	showBreakdown();
}

void MainWindow::showBreakdown()
{
	breakdown_->clear();
	const int g = selectedGuess();
	if (!shown_ || g < 0)
	{
		breakdown_title_->clear();
		return;
	}
	Engine * eng = shown_eng_;
	std::vector<std::vector<int>> groups(NPAT);
	for (uint16_t c : shown_->cands) groups[eng->pat(g, c)].push_back(eng->pool()[c]);
	std::vector<int> order;
	for (int p = 0; p < NPAT; p++)
		if (!groups[p].empty()) order.push_back(p);
	std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return groups[x].size() > groups[y].size(); });

	breakdown_title_->setText(tr("<b>%1</b> splits the %2 possible answers into %3 groups:")
		.arg(QString::fromStdString(eng->words()[g]).toUpper()).arg(shown_->cands.size()).arg(order.size()));
	for (int p : order)
	{
		QStringList ws;
		for (size_t i = 0; i < groups[p].size() && i < 40; i++) ws << QString::fromStdString(eng->words()[groups[p][i]]);
		if (groups[p].size() > 40) ws << QStringLiteral("…");
		auto * it = new QListWidgetItem(QIcon(colors::pattern_icon(p, 14)),
			QStringLiteral("%1   %2").arg(groups[p].size(), 4).arg(ws.join(' ')));
		breakdown_->addItem(it);
	}
}
