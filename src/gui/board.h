// The 6x5 tile grid and on-screen keyboard. Type a guess, click tiles to set the
// colours the game showed, press Enter to add the row.
#pragma once

#include <QWidget>

#include <functional>

class Board : public QWidget
{
	Q_OBJECT
public:
	struct Row
	{
		QString word;
		int digit[5] = { 0, 0, 0, 0, 0 }; // 0 gray, 1 yellow, 2 green
		int pattern() const;
	};

	explicit Board(QWidget * parent = nullptr);

	// Rows added with Enter, oldest first.
	QList<Row> turns() const { return rows_.mid(0, committed_); }
	bool solved() const;

	// Returns an error message for a word that may not be played, or an empty string.
	void setValidator(std::function<QString(const QString &)> v) { validator_ = std::move(v); }
	void setEditWord(const QString & word);
	Q_INVOKABLE void undo();
	Q_INVOKABLE void reset();
	// Replaces the committed rows without checking them (for replaying a saved game).
	void setTurns(const QList<Row> & rows);

	QSize sizeHint() const override;
	QSize minimumSizeHint() const override { return sizeHint(); }

signals:
	void turnsChanged();
	void rejected(const QString & why);

protected:
	void paintEvent(QPaintEvent *) override;
	void mousePressEvent(QMouseEvent * e) override;
	void keyPressEvent(QKeyEvent * e) override;

private:
	struct Key { QRectF rect; QString label; int code; };

	void typeKey(int code);
	void submit();
	QRectF tileRect(int row, int col) const;
	QList<Key> keys() const;
	int letterState(QChar c) const; // -1 unknown, else best digit seen

	QList<Row> rows_; // committed rows, then the row being edited
	int committed_ = 0;
	std::function<QString(const QString &)> validator_;
};
