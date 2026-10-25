#include "gamesdialog.h"
#include "colors.h"
#include "game.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>

GamesDialog::GamesDialog(QWidget * parent) : QDialog(parent), db_(std::make_unique<GamesDb>())
{
	setWindowTitle(tr("Saved games"));

	source_ = new QComboBox;
	for (const GameSource & s : db_->sources())
		if (s.with_words)
			source_->addItem(tr("%1 — %2 games").arg(QString::fromStdString(s.name)).arg(s.with_words),
				QString::fromStdString(s.name));
	answer_ = new QLineEdit;
	answer_->setPlaceholderText(tr("Answer"));
	answer_->setMaxLength(5);
	answer_->setClearButtonEnabled(true);
	consistent_ = new QCheckBox(tr("Real colours only"));
	consistent_->setChecked(true);
	consistent_->setToolTip(tr("Some sources coloured repeated letters their own way.\n"
		"Their games can reach positions the real game can't."));
	auto * find = new QPushButton(tr("Find"));
	auto * random = new QPushButton(tr("Random"));
	auto * filters = new QHBoxLayout;
	filters->addWidget(source_, 1);
	filters->addWidget(answer_);
	filters->addWidget(consistent_);
	filters->addWidget(find);
	filters->addWidget(random);

	table_ = new QTableWidget(0, 5);
	table_->setHorizontalHeaderLabels({ tr("Game"), tr("Answer"), tr("Result"), tr("Player"), tr("Guesses") });
	table_->setSelectionBehavior(QAbstractItemView::SelectRows);
	table_->setSelectionMode(QAbstractItemView::SingleSelection);
	table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
	table_->verticalHeader()->hide();
	table_->horizontalHeader()->setStretchLastSection(true);
	table_->setIconSize(colors::pattern_icon(0, 10).size());

	info_ = new QLabel(tr("Only games whose guessed words are known can be replayed. These come "
		"from other people's datasets: keep them on this computer."));
	info_->setWordWrap(true);
	info_->setEnabled(false);

	auto * buttons = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
	auto * lv = new QVBoxLayout(this);
	lv->addLayout(filters);
	lv->addWidget(table_);
	lv->addWidget(info_);
	lv->addWidget(buttons);

	connect(find, &QPushButton::clicked, this, [this] { search(false); });
	connect(random, &QPushButton::clicked, this, [this] { search(true); });
	connect(answer_, &QLineEdit::returnPressed, this, [this] { search(false); });
	connect(source_, &QComboBox::currentIndexChanged, this, [this] { search(false); });
	connect(table_, &QTableWidget::cellDoubleClicked, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
	source_->setObjectName("source");
	random->setObjectName("random");
	table_->setObjectName("games");

	resize(900, 560);
	search(false);
}

GamesDialog::~GamesDialog() = default;

StoredGame GamesDialog::selected() const
{
	const int r = table_->currentRow();
	return r >= 0 && r < (int)games_.size() ? games_[r] : StoredGame();
}

void GamesDialog::search(bool random)
{
	GameFilter f;
	f.source = source_->currentData().toString().toStdString();
	f.answer = answer_->text().trimmed().toLower().toStdString();
	f.words_only = true;
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
		info_->setText(QString::fromStdString(e.what()));
	}
	table_->setRowCount((int)games_.size());
	for (int i = 0; i < (int)games_.size(); i++)
	{
		const StoredGame & g = games_[i];
		auto item = [&](int col, const QString & text) { table_->setItem(i, col, new QTableWidgetItem(text)); };
		item(0, QString::number(g.id));
		item(1, g.answer.empty() ? QStringLiteral("?") : QString::fromStdString(g.answer).toUpper());
		item(2, (g.result > 0 ? QString("%1/6").arg(g.result) : g.result == 0 ? QStringLiteral("X/6") : tr("unfinished")) +
			(g.hard == 1 ? QStringLiteral("*") : QString()));
		item(3, QString::fromStdString(g.player.empty() ? g.variant : g.player));
		QStringList words;
		for (const auto & w : g.guesses) words << QString::fromStdString(w).toUpper();
		auto * guesses = new QTableWidgetItem(words.join(' '));
		if (!g.colors.empty())
			if (auto pat = parse_pattern(g.colors.back())) guesses->setIcon(QIcon(colors::pattern_icon(*pat, 10)));
		table_->setItem(i, 4, guesses);
	}
	table_->resizeColumnsToContents();
	if (!games_.empty()) table_->selectRow(0);
}
