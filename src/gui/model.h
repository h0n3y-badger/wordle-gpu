// Table of ranked guesses for the current position.
#pragma once

#include "game.h"

#include <QAbstractTableModel>
#include <QSortFilterProxyModel>

#include <memory>

class SuggestionModel : public QAbstractTableModel
{
	Q_OBJECT
public:
	enum Column { RANK, GUESS, SCORE, WIN_BY, BITS, GROUPS, AVG_LEFT, WORST, NCOLS };
	static constexpr int SortRole = Qt::UserRole;
	static constexpr int GuessRole = Qt::UserRole + 1;
	static constexpr int CandidateRole = Qt::UserRole + 2;

	using QAbstractTableModel::QAbstractTableModel;

	void set(std::shared_ptr<const Analysis> a, const Engine * eng);
	const Analysis * analysis() const { return a_.get(); }

	int rowCount(const QModelIndex & = {}) const override { return a_ ? (int)a_->rows.size() : 0; }
	int columnCount(const QModelIndex & = {}) const override { return NCOLS; }
	QVariant data(const QModelIndex & idx, int role) const override;
	QVariant headerData(int section, Qt::Orientation o, int role) const override;

private:
	std::shared_ptr<const Analysis> a_;
	const Engine * eng_ = nullptr;
};

// Filters by "possible answers only" and a word prefix.
class SuggestionFilter : public QSortFilterProxyModel
{
	Q_OBJECT
public:
	using QSortFilterProxyModel::QSortFilterProxyModel;
	void setOnlyCandidates(bool on) { beginFilterChange(); only_ = on; endFilterChange(); }
	void setPrefix(const QString & s) { beginFilterChange(); prefix_ = s.toLower(); endFilterChange(); }

protected:
	bool filterAcceptsRow(int row, const QModelIndex & parent) const override;

private:
	bool only_ = false;
	QString prefix_;
};
