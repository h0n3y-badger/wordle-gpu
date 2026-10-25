// Pick a game from the games database to replay on the board.
#pragma once

#include "gamesdb.h"

#include <QDialog>

#include <memory>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QTableWidget;

class GamesDialog : public QDialog
{
	Q_OBJECT
public:
	// Throws if the database can't be opened.
	explicit GamesDialog(QWidget * parent = nullptr);
	~GamesDialog() override;

	StoredGame selected() const;

private:
	void search(bool random);

	std::unique_ptr<GamesDb> db_;
	std::vector<StoredGame> games_;
	QComboBox * source_;
	QLineEdit * answer_;
	QCheckBox * consistent_;
	QTableWidget * table_;
	QLabel * info_;
};
