#include "board.h"
#include "colors.h"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>

namespace {
constexpr int TILE = 56, GAP = 6, MARGIN = 8;
constexpr int KEY_W = 30, KEY_H = 42, KEY_GAP = 4;
constexpr int GRID_W = 5 * TILE + 4 * GAP, GRID_H = 6 * TILE + 5 * GAP;
constexpr int KB_W = 10 * KEY_W + 9 * KEY_GAP, KB_H = 3 * KEY_H + 2 * KEY_GAP;
constexpr int WIDTH = KB_W + 2 * MARGIN, HEIGHT = GRID_H + 20 + KB_H + 2 * MARGIN;
constexpr int KEY_ENTER = Qt::Key_Return, KEY_BACK = Qt::Key_Backspace;
}

int Board::Row::pattern() const
{
	int p = 0;
	for (int i = 4; i >= 0; i--) p = p * 3 + digit[i];
	return p;
}

Board::Board(QWidget * parent) : QWidget(parent)
{
	setFocusPolicy(Qt::StrongFocus);
	reset();
}

QSize Board::sizeHint() const { return QSize(WIDTH, HEIGHT); }

bool Board::solved() const
{
	return committed_ > 0 && rows_[committed_ - 1].pattern() == 242;
}

void Board::reset()
{
	rows_ = { Row() };
	committed_ = 0;
	update();
	emit turnsChanged();
}

void Board::undo()
{
	if (committed_ == 0) return;
	if (committed_ < rows_.size()) rows_.removeLast();
	committed_--;
	update();
	emit turnsChanged();
}

void Board::setEditWord(const QString & word)
{
	if (committed_ >= 6 || solved()) return;
	Row & r = rows_[committed_];
	r.word = word.left(5).toLower();
	for (int & d : r.digit) d = 0;
	update();
}

QRectF Board::tileRect(int row, int col) const
{
	const qreal x0 = (width() - GRID_W) / 2.0, y0 = MARGIN;
	return QRectF(x0 + col * (TILE + GAP), y0 + row * (TILE + GAP), TILE, TILE);
}

QList<Board::Key> Board::keys() const
{
	static const char * layout[3] = { "qwertyuiop", "asdfghjkl", "zxcvbnm" };
	QList<Key> out;
	const qreal y0 = MARGIN + GRID_H + 20;
	for (int r = 0; r < 3; r++)
	{
		const int n = (int)strlen(layout[r]);
		const qreal wide = r == 2 ? 1.5 * KEY_W + KEY_GAP / 2.0 : 0; // enter and backspace
		const qreal w = n * KEY_W + (n - 1) * KEY_GAP + (r == 2 ? 2 * (wide + KEY_GAP) : 0);
		qreal x = (width() - w) / 2.0;
		const qreal y = y0 + r * (KEY_H + KEY_GAP);
		if (r == 2)
		{
			out.push_back({ QRectF(x, y, wide, KEY_H), "Enter", KEY_ENTER });
			x += wide + KEY_GAP;
		}
		for (int i = 0; i < n; i++)
		{
			out.push_back({ QRectF(x, y, KEY_W, KEY_H), QString(QChar(layout[r][i])).toUpper(), Qt::Key_A + layout[r][i] - 'a' });
			x += KEY_W + KEY_GAP;
		}
		if (r == 2) out.push_back({ QRectF(x, y, wide, KEY_H), QString::fromUtf8("←"), KEY_BACK });
	}
	return out;
}

int Board::letterState(QChar c) const
{
	int best = -1;
	for (int r = 0; r < committed_; r++)
		for (int i = 0; i < 5; i++)
			if (rows_[r].word[i] == c) best = std::max(best, rows_[r].digit[i]);
	return best;
}

void Board::paintEvent(QPaintEvent *)
{
	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing);
	QFont tileFont = font();
	tileFont.setPixelSize(28);
	tileFont.setBold(true);
	QColor border = palette().color(QPalette::Text);
	border.setAlphaF(0.3);

	for (int r = 0; r < 6; r++)
	{
		for (int c = 0; c < 5; c++)
		{
			const QRectF rect = tileRect(r, c).adjusted(1, 1, -1, -1);
			const bool has = r < rows_.size() && c < rows_[r].word.size();
			if (has)
			{
				const QColor fill = colors::digit(rows_[r].digit[c]);
				p.setPen(r == committed_ ? QPen(palette().color(QPalette::Highlight), 2) : Qt::NoPen);
				p.setBrush(fill);
				p.drawRoundedRect(rect, 3, 3);
				p.setPen(Qt::white);
				p.setFont(tileFont);
				p.drawText(rect, Qt::AlignCenter, rows_[r].word.mid(c, 1).toUpper());
			}
			else
			{
				const bool editing = r == committed_ && r < rows_.size();
				p.setPen(QPen(editing ? palette().color(QPalette::Text) : border, editing ? 1.5 : 1));
				p.setBrush(Qt::NoBrush);
				p.drawRoundedRect(rect, 3, 3);
			}
		}
	}

	QFont keyFont = font();
	keyFont.setPixelSize(14);
	keyFont.setBold(true);
	p.setFont(keyFont);
	for (const Key & k : keys())
	{
		int state = -1;
		if (k.code >= Qt::Key_A && k.code <= Qt::Key_Z) state = letterState(QChar('a' + k.code - Qt::Key_A));
		p.setPen(Qt::NoPen);
		p.setBrush(state < 0 ? colors::KEY : colors::digit(state));
		p.drawRoundedRect(k.rect, 4, 4);
		p.setPen(Qt::white);
		p.drawText(k.rect, Qt::AlignCenter, k.label);
	}
}

void Board::mousePressEvent(QMouseEvent * e)
{
	setFocus();
	const QPointF pos = e->position();
	for (int r = 0; r <= committed_ && r < rows_.size(); r++)
		for (int c = 0; c < 5; c++)
			if (tileRect(r, c).contains(pos) && c < rows_[r].word.size())
			{
				int & d = rows_[r].digit[c];
				d = (d + (e->button() == Qt::RightButton ? 2 : 1)) % 3;
				update();
				if (r < committed_) emit turnsChanged();
				return;
			}
	for (const Key & k : keys())
		if (k.rect.contains(pos)) { typeKey(k.code); return; }
}

void Board::keyPressEvent(QKeyEvent * e)
{
	const int k = e->key();
	if ((k >= Qt::Key_A && k <= Qt::Key_Z) || k == Qt::Key_Return || k == Qt::Key_Enter || k == Qt::Key_Backspace)
	{
		if (e->modifiers() & (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier))
			return QWidget::keyPressEvent(e);
		typeKey(k == Qt::Key_Enter ? KEY_ENTER : k);
		return;
	}
	// 1-5 cycle the colour of that tile in the row being edited.
	if (k >= Qt::Key_1 && k <= Qt::Key_5 && committed_ < rows_.size())
	{
		const int c = k - Qt::Key_1;
		if (c < rows_[committed_].word.size())
		{
			int & d = rows_[committed_].digit[c];
			d = (d + 1) % 3;
			update();
		}
		return;
	}
	QWidget::keyPressEvent(e);
}

void Board::typeKey(int code)
{
	if (committed_ >= rows_.size()) return;
	Row & r = rows_[committed_];
	if (code == KEY_ENTER) { submit(); return; }
	if (code == KEY_BACK)
	{
		if (!r.word.isEmpty())
		{
			r.digit[r.word.size() - 1] = 0;
			r.word.chop(1);
		}
	}
	else if (r.word.size() < 5)
		r.word += QChar('a' + code - Qt::Key_A);
	update();
}

void Board::submit()
{
	Row & r = rows_[committed_];
	if (r.word.size() < 5) { emit rejected(tr("Type all five letters first.")); return; }
	if (validator_)
	{
		const QString why = validator_(r.word);
		if (!why.isEmpty()) { emit rejected(why); return; }
	}
	committed_++;
	if (committed_ < 6 && !solved()) rows_.push_back(Row());
	update();
	emit turnsChanged();
}
