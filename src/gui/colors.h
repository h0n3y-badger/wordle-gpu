// The game's tile colours.
#pragma once

#include <QColor>
#include <QPainter>
#include <QPixmap>

namespace colors {

inline const QColor GREEN(0x53, 0x8d, 0x4e);
inline const QColor YELLOW(0xb5, 0x9f, 0x3b);
inline const QColor GRAY(0x3a, 0x3a, 0x3c);
inline const QColor KEY(0x81, 0x83, 0x84);

inline QColor digit(int d) { return d == 2 ? GREEN : d == 1 ? YELLOW : GRAY; }

// Five little squares showing a feedback pattern.
inline QPixmap pattern_icon(int pat, int size = 12)
{
	QPixmap pm(5 * size + 4 * 2, size);
	pm.fill(Qt::transparent);
	QPainter p(&pm);
	for (int i = 0; i < 5; i++, pat /= 3)
		p.fillRect(i * (size + 2), 0, size, size, digit(pat % 3));
	return pm;
}

}
