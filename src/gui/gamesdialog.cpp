#include "gamesdialog.h"
#include "colors.h"
#include "game.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QIntValidator>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextBrowser>
#include <QTextDocument>
#include <QUrl>
#include <QVBoxLayout>

namespace {

// All of a game's colour rows as one small picture.
QPixmap grid_icon(const std::vector<std::string> & rows, int size = 4)
{
	QPixmap pm(5 * size + 4, std::max<int>(1, (int)rows.size()) * (size + 1) - 1);
	pm.fill(Qt::transparent);
	QPainter p(&pm);
	for (size_t r = 0; r < rows.size(); r++)
		for (int i = 0; i < 5 && i < (int)rows[r].size(); i++)
		{
			const char c = rows[r][i];
			p.fillRect(i * (size + 1), (int)r * (size + 1), size, size,
				colors::digit(c == 'g' ? 2 : c == 'y' ? 1 : 0));
		}
	return pm;
}

// A row of tiles in rich text, with letters if the word is known.
QString tiles_html(const std::string & word, const std::string & row)
{
	QString s = QStringLiteral("<table cellspacing='2' cellpadding='4'><tr>");
	for (int i = 0; i < 5 && i < (int)row.size(); i++)
	{
		const QColor c = colors::digit(row[i] == 'g' ? 2 : row[i] == 'y' ? 1 : 0);
		s += QStringLiteral("<td bgcolor='%1' align='center' width='26'><b><font color='white'>%2</font></b></td>")
			.arg(c.name(), word.empty() ? QStringLiteral("&nbsp;") : QString(QChar(word[i]).toUpper()));
	}
	return s + QStringLiteral("</tr></table>");
}

QString result_text(const StoredGame & g)
{
	return (g.result > 0 ? QString("%1/6").arg(g.result) : g.result == 0 ? QStringLiteral("X/6")
			: GamesDialog::tr("unfinished")) + (g.hard == 1 ? QStringLiteral("*") : QString());
}

}

GamesDialog::GamesDialog(const std::vector<std::string> & allowed, QWidget * parent)
	: QDialog(parent), db_(std::make_unique<GamesDb>()), allowed_(allowed)
{
	setWindowTitle(tr("Saved games"));
	const auto sources = db_->sources();

	// Games: every source; the ones with words can be replayed.
	auto * games_page = new QWidget;
	source_ = new QComboBox;
	for (const GameSource & s : sources)
	{
		if (!s.games) continue;
		const QString label = s.with_words
			? tr("%1 — %L2 games with words").arg(QString::fromStdString(s.name)).arg(s.games)
			: tr("%1 — %L2 games, colours only").arg(QString::fromStdString(s.name)).arg(s.games);
		source_->addItem(label, QString::fromStdString(s.name));
		source_->setItemData(source_->count() - 1, QString::fromStdString(s.title + "\n" + s.notes), Qt::ToolTipRole);
	}
	answer_ = new QLineEdit;
	answer_->setPlaceholderText(tr("Answer"));
	answer_->setMaxLength(5);
	answer_->setClearButtonEnabled(true);
	answer_->setMaximumWidth(110);
	puzzle_filter_ = new QLineEdit;
	puzzle_filter_->setPlaceholderText(tr("Puzzle #"));
	puzzle_filter_->setValidator(new QIntValidator(0, 99999, puzzle_filter_));
	puzzle_filter_->setClearButtonEnabled(true);
	puzzle_filter_->setMaximumWidth(110);
	consistent_ = new QCheckBox(tr("Real colours only"));
	consistent_->setChecked(true);
	consistent_->setToolTip(tr("Some sources coloured repeated letters their own way.\n"
		"Their games can reach positions the real game can't."));
	auto * find = new QPushButton(tr("Find"));
	auto * random = new QPushButton(tr("Random"));
	auto * filters = new QHBoxLayout;
	filters->addWidget(source_, 1);
	filters->addWidget(answer_);
	filters->addWidget(puzzle_filter_);
	filters->addWidget(consistent_);
	filters->addWidget(find);
	filters->addWidget(random);

	table_ = new QTableWidget(0, 6);
	table_->setHorizontalHeaderLabels({ tr("Game"), tr("Puzzle"), tr("Answer"), tr("Result"), tr("Player"), tr("Guesses") });
	table_->setSelectionBehavior(QAbstractItemView::SelectRows);
	table_->setSelectionMode(QAbstractItemView::SingleSelection);
	table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
	table_->verticalHeader()->hide();
	table_->horizontalHeader()->setStretchLastSection(true);
	table_->setIconSize(grid_icon(std::vector<std::string>(6, "bbbbb")).size());
	table_->verticalHeader()->setDefaultSectionSize(grid_icon(std::vector<std::string>(6, "bbbbb")).height() + 6);
	details_ = new QTextBrowser;
	details_->setOpenLinks(false);
	auto * split = new QSplitter(Qt::Vertical);
	split->addWidget(table_);
	split->addWidget(details_);
	split->setStretchFactor(0, 3);
	split->setStretchFactor(1, 2);
	auto * gv = new QVBoxLayout(games_page);
	gv->addLayout(filters);
	gv->addWidget(split);

	// Puzzles: what every source says about one NYT puzzle.
	auto * puzzles_page = new QWidget;
	puzzle_ = new QSpinBox;
	const auto [lo, hi] = db_->daily_puzzles();
	puzzle_->setRange(std::max(0, lo), std::max(0, hi));
	puzzle_->setPrefix(tr("Wordle "));
	puzzle_title_ = new QLabel;
	auto * its_games = new QPushButton(tr("Its games"));
	its_games->setToolTip(tr("List the games played on this puzzle."));
	auto * prow = new QHBoxLayout;
	prow->addWidget(puzzle_);
	prow->addWidget(puzzle_title_, 1);
	prow->addWidget(its_games);
	report_ = new QTextBrowser;
	report_->setOpenLinks(false);
	auto * pv = new QVBoxLayout(puzzles_page);
	pv->addLayout(prow);
	pv->addWidget(report_);

	tabs_ = new QTabWidget;
	tabs_->addTab(games_page, tr("Games"));
	tabs_->addTab(puzzles_page, tr("Puzzles"));
	if (lo < 0) tabs_->setTabEnabled(1, false);

	auto * info = new QLabel(tr("Games with words open on the board to replay. These come from other "
		"people's datasets: keep them on this computer."));
	info->setWordWrap(true);
	info->setEnabled(false);
	auto * buttons = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Close);
	open_ = buttons->button(QDialogButtonBox::Open);
	auto * lv = new QVBoxLayout(this);
	lv->addWidget(tabs_);
	lv->addWidget(info);
	lv->addWidget(buttons);

	connect(find, &QPushButton::clicked, this, [this] { search(false); });
	connect(random, &QPushButton::clicked, this, [this] { search(true); });
	connect(answer_, &QLineEdit::returnPressed, this, [this] { search(false); });
	connect(puzzle_filter_, &QLineEdit::returnPressed, this, [this] { search(false); });
	connect(source_, &QComboBox::currentIndexChanged, this, [this] { search(false); });
	connect(consistent_, &QCheckBox::toggled, this, [this] { search(false); });
	connect(table_, &QTableWidget::itemSelectionChanged, this, &GamesDialog::showGame);
	connect(table_, &QTableWidget::cellDoubleClicked, this, [this] { if (open_->isEnabled()) accept(); });
	connect(puzzle_, &QSpinBox::valueChanged, this, &GamesDialog::showPuzzle);
	connect(its_games, &QPushButton::clicked, this, [this] {
		puzzle_filter_->setText(QString::number(puzzle_->value()));
		answer_->clear();
		// Shared results are the games that have puzzle numbers.
		for (int i = 0; i < source_->count(); i++)
			if (source_->itemData(i).toString().startsWith("tweets-")) { source_->setCurrentIndex(i); break; }
		search(false);
		tabs_->setCurrentIndex(0);
	});
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
	source_->setObjectName("source");
	random->setObjectName("random");
	tabs_->setObjectName("tabs");
	table_->setObjectName("games");

	resize(1000, 720);
	puzzle_->setValue(puzzle_->maximum());
	showPuzzle(puzzle_->value());
	search(false);
}

GamesDialog::~GamesDialog() = default;

StoredGame GamesDialog::selected() const
{
	const int r = table_->currentRow();
	return r >= 0 && r < (int)games_.size() && !games_[r].guesses.empty() ? games_[r] : StoredGame();
}

void GamesDialog::search(bool random)
{
	GameFilter f;
	f.source = source_->currentData().toString().toStdString();
	f.answer = answer_->text().trimmed().toLower().toStdString();
	if (!puzzle_filter_->text().isEmpty()) f.puzzle = puzzle_filter_->text().toInt();
	f.consistent = consistent_->isChecked();
	f.random = random;
	f.limit = 200;
	try
	{
		games_ = db_->games(f);
	}
	catch (const std::exception & e)
	{
		games_.clear();
		details_->setPlainText(QString::fromStdString(e.what()));
	}
	table_->setRowCount((int)games_.size());
	for (int i = 0; i < (int)games_.size(); i++)
	{
		const StoredGame & g = games_[i];
		auto item = [&](int col, const QString & text) { table_->setItem(i, col, new QTableWidgetItem(text)); };
		item(0, QString::number(g.id));
		item(1, g.puzzle >= 0 ? QString::number(g.puzzle) : QString());
		item(2, g.answer.empty() ? QStringLiteral("?") : QString::fromStdString(g.answer).toUpper());
		item(3, result_text(g));
		item(4, QString::fromStdString(g.player.empty() ? g.variant : g.player));
		QStringList words;
		for (const auto & w : g.guesses) words << QString::fromStdString(w).toUpper();
		auto * guesses = new QTableWidgetItem(QIcon(grid_icon(g.colors)),
			words.isEmpty() ? tr("(colours only)") : words.join(' '));
		table_->setItem(i, 5, guesses);
	}
	table_->resizeColumnsToContents();
	if (!games_.empty()) table_->selectRow(0);
	else
	{
		details_->setHtml(tr("No games match."));
		open_->setEnabled(false);
	}
}

// The selected game in full. Colour-only rows list the guesses that give them.
void GamesDialog::showGame()
{
	const int r = table_->currentRow();
	if (r < 0 || r >= (int)games_.size())
	{
		open_->setEnabled(false);
		return;
	}
	const StoredGame & g = games_[r];
	open_->setEnabled(!g.guesses.empty());

	QString h = tr("<b>Game %1</b> from %2").arg(g.id).arg(QString::fromStdString(g.source));
	if (g.puzzle >= 0) h += tr(", Wordle %1 (%2)").arg(g.puzzle).arg(QString::fromStdString(g.date()));
	h += tr(" — answer <b>%1</b>, %2").arg(g.answer.empty() ? QStringLiteral("?") : QString::fromStdString(g.answer).toUpper(),
		result_text(g));
	if (g.hard == 1) h += tr(", hard mode");
	if (!g.player.empty()) h += tr(", player %1").arg(QString::fromStdString(g.player).toHtmlEscaped());
	if (!g.variant.empty()) h += QStringLiteral(" (%1)").arg(QString::fromStdString(g.variant).toHtmlEscaped());
	if (g.consistent == 0)
		h += tr("<br><i>These colours don't fit the answer under the real rules: the source coloured "
			"repeated letters its own way.</i>");

	h += QStringLiteral("<table cellspacing='0'>");
	for (size_t i = 0; i < g.colors.size(); i++)
	{
		const std::string & row = g.colors[i];
		h += QStringLiteral("<tr><td>") + tiles_html(i < g.guesses.size() ? g.guesses[i] : std::string(), row) +
			QStringLiteral("</td><td valign='middle'>&nbsp;&nbsp;");
		if (g.guesses.empty() && !g.answer.empty() && g.answer.size() == 5)
		{
			// Which guesses get these colours against the answer.
			const auto want = parse_pattern(row);
			QStringList fits;
			int n = 0;
			for (const auto & w : allowed_)
				if (want && feedback(w.data(), g.answer.data()) == *want && n++ < 12)
					fits << QString::fromStdString(w);
			if (n == 0) h += tr("no word gives these colours");
			else
				h += (n == 1 ? tr("1 word gives these colours: ") : tr("%L1 words give these colours: ").arg(n)) +
					fits.join(' ') + (n > 12 ? QStringLiteral(" …") : QString());
		}
		h += QStringLiteral("</td></tr>");
	}
	h += QStringLiteral("</table>");
	h += g.guesses.empty()
		? tr("Only the colours were shared, so this game can't be replayed on the board.")
		: tr("Open (or double-click) replays it on the board, a guess at a time.");
	details_->setHtml(h);
}

// Every source's spread of guess counts for one puzzle, as bars.
void GamesDialog::showPuzzle(int puzzle)
{
	StoredGame probe;
	probe.puzzle = puzzle;
	const std::string answer = db_->answer(puzzle);
	puzzle_title_->setText(tr("%1 — answer <b>%2</b>").arg(QString::fromStdString(probe.date()),
		answer.empty() ? tr("unknown") : QString::fromStdString(answer).toUpper()));

	QString h;
	const auto stats = db_->day_stats(puzzle);
	// Rich text ignores relative widths in nested tables, so the bars are pictures.
	const int bar_w = std::max(200, report_->viewport()->width() - 160);
	auto bar = [&](int key, double frac, const QColor & c) {
		QPixmap pm(std::max(2, (int)std::lround(bar_w * frac)), 14);
		pm.fill(c);
		const QUrl url(QStringLiteral("bar:%1").arg(key));
		report_->document()->addResource(QTextDocument::ImageResource, url, pm);
		return QStringLiteral("<img src='%1'>").arg(url.toString());
	};
	int bars = 0;
	for (const DayStats & d : stats)
	{
		double sum = 0, top = 0;
		for (double x : d.n) { sum += x; top = std::max(top, x); }
		h += tr("<p><b>%1</b> %2: %L3 %4").arg(QString::fromStdString(d.source), QString::fromStdString(d.mode))
			.arg(d.games).arg(d.mode == "shared" ? tr("shared results") : tr("players"));
		if (d.hard_games >= 0) h += tr(" (%L1 in hard mode)").arg(d.hard_games);
		if (d.average >= 0)
			h += tr(", average %1%2").arg(d.average, 0, 'f', 3).arg(d.average_solved ? tr(" when solved") : QString());
		if (sum <= 0)
		{
			h += tr("<br><i>no spread of guesses for this day</i></p>");
			continue;
		}
		h += QStringLiteral("</p><table cellspacing='0' cellpadding='1'>");
		for (int i = 0; i < 7; i++)
			h += QStringLiteral("<tr><td align='right'>%1&nbsp;</td><td>%2</td><td>&nbsp;%3%</td></tr>")
				.arg(i == 6 ? QStringLiteral("X") : QString::number(i + 1))
				.arg(bar(bars++, d.n[i] / top, i == 6 ? colors::KEY : colors::GREEN))
				.arg(100 * d.n[i] / sum, 0, 'f', 1);
		h += QStringLiteral("</table>");
	}
	if (stats.empty()) h = tr("No data for this puzzle.");
	else
		h += tr("<p><i>WordleBot counts players who used it that day; the -ps modes are undocumented. "
			"The MCM data gives percentages of results reported on Twitter in 2022. Shared results "
			"are the tweet datasets' games.</i></p>");
	report_->setHtml(h);
}
