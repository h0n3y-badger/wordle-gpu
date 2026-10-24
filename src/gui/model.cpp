#include "model.h"
#include "exact.h"

#include <QFont>

void SuggestionModel::set(std::shared_ptr<const Analysis> a, const Engine * eng)
{
	beginResetModel();
	a_ = std::move(a);
	eng_ = eng;
	endResetModel();
}

QVariant SuggestionModel::data(const QModelIndex & idx, int role) const
{
	if (!a_ || !idx.isValid()) return {};
	const Analysis & a = *a_;
	const Suggestion & s = a.rows[idx.row()];
	const double n = (double)a.cands.size();
	const bool deep = s.la.by3 >= 0;

	if (role == GuessRole) return s.guess;
	if (role == CandidateRole) return (bool)s.st.candidate;
	if (role == Qt::TextAlignmentRole)
		return idx.column() == GUESS ? QVariant(Qt::AlignLeft | Qt::AlignVCenter) : QVariant(Qt::AlignRight | Qt::AlignVCenter);
	if (role == Qt::FontRole && idx.column() == GUESS && s.st.candidate)
	{
		QFont f;
		f.setBold(true);
		return f;
	}
	if (role == Qt::ToolTipRole)
	{
		switch (idx.column())
		{
			case GUESS: return s.st.candidate ? tr("Could be the answer") : tr("Cannot be the answer, but narrows things down");
			case SCORE:
				if (s.exact >= INF) return tr("No way to finish within %1 guesses after this").arg(MAX_GUESSES);
				if (s.exact >= 0) return tr("Exact: best possible average with perfect play from here");
				return tr("Lower bound from trying every second guess; the exact value is at least this");
			default: return {};
		}
	}

	const bool sort = role == SortRole;
	if (role != Qt::DisplayRole && !sort) return {};
	switch (idx.column())
	{
		case RANK: return sort ? QVariant(idx.row()) : QVariant(idx.row() + 1);
		case GUESS:
			return QString::fromStdString(eng_->words()[s.guess]) + (s.st.candidate && !sort ? QStringLiteral("  ●") : QString());
		case SCORE:
			if (sort) return s.exact >= 0 ? s.exact : s.la.total;
			if (s.exact >= INF) return tr("fails");
			if (s.exact >= 0) return QString::number(a.expected_exact(s), 'f', 4);
			if (deep) return QStringLiteral("≥ ") + QString::number(a.expected_lb(s), 'f', 4);
			return QString();
		case WIN_BY:
			if (!deep) return sort ? QVariant(-1) : QVariant(QString());
			return sort ? QVariant(s.la.by3) : QVariant(QString::number(100.0 * s.la.by3 / n, 'f', 1) + '%');
		case BITS: return sort ? QVariant(s.st.entropy) : QVariant(QString::number(s.st.entropy, 'f', 3));
		case GROUPS: return s.st.buckets;
		case AVG_LEFT: return sort ? QVariant(s.st.sum_sq / n) : QVariant(QString::number(s.st.sum_sq / n, 'f', 2));
		case WORST: return s.st.largest;
	}
	return {};
}

QVariant SuggestionModel::headerData(int section, Qt::Orientation o, int role) const
{
	if (o != Qt::Horizontal) return {};
	const int by = (a_ ? a_->played : 0) + 3;
	if (role == Qt::DisplayRole)
	{
		switch (section)
		{
			case RANK: return tr("#");
			case GUESS: return tr("Guess");
			case SCORE: return tr("Avg score");
			case WIN_BY: return tr("Solved by %1").arg(by);
			case BITS: return tr("Bits");
			case GROUPS: return tr("Groups");
			case AVG_LEFT: return tr("Avg left");
			case WORST: return tr("Worst");
		}
	}
	if (role == Qt::ToolTipRole)
	{
		switch (section)
		{
			case SCORE: return tr("Expected guess number the game ends on, playing this next and perfectly after.\n"
				"\"≥\" values come from the GPU brute force over every second guess; plain values are exact.");
			case WIN_BY: return tr("Share of possible answers solved by guess %1, with the best follow-up for each colour result").arg(by);
			case BITS: return tr("Information gained on average (entropy of the colour results)");
			case GROUPS: return tr("Number of different colour results this guess can get");
			case AVG_LEFT: return tr("Expected number of possible answers left after this guess");
			case WORST: return tr("Possible answers left in the worst case");
		}
	}
	return {};
}

bool SuggestionFilter::filterAcceptsRow(int row, const QModelIndex & parent) const
{
	const QModelIndex idx = sourceModel()->index(row, SuggestionModel::GUESS, parent);
	if (only_ && !idx.data(SuggestionModel::CandidateRole).toBool()) return false;
	return prefix_.isEmpty() || idx.data(Qt::DisplayRole).toString().startsWith(prefix_);
}
