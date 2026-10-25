// wordle-gpu: GUI front end.
#include "mainwindow.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QKeyEvent>
#include <QMessageBox>
#include <QPushButton>
#include <QTabWidget>
#include <QTableView>
#include <QTimer>

// WORDLE_GPU_TEST drives the window for automated checks, e.g. with QT_QPA_PLATFORM=offscreen:
// space-separated steps, each typed letters/digits (sent to the board as key presses),
// "ret", "bs", "wait" (until the analysis is done), "sleep:MS", "exact", "stop", "undo",
// "hard", "pool", "row:N",
// "tab:N", "replay:GAME_ID", "next", "prev", "games" (the saved-games dialog), "dlg:random",
// "dlg:tab:N", "dlg:source:N", "dlg:row:N", "dlg:open", "dlg:shot:FILE", "shot:FILE" or "quit".
static void run_test(MainWindow * w, QStringList list)
{
	auto steps = std::make_shared<QStringList>(list);
	auto * board = w->findChild<QWidget *>("board");
	auto * stop = w->findChild<QPushButton *>("stop");
	auto step = std::make_shared<std::function<void()>>();
	*step = [=] {
		if (steps->isEmpty()) return;
		const QString s = steps->takeFirst();
		int delay = 30;
		auto key = [&](int k, const QString & text) {
			QKeyEvent press(QEvent::KeyPress, k, Qt::NoModifier, text);
			QApplication::sendEvent(board, &press);
		};
		if (s == "wait")
		{
			if (stop->isEnabled()) { steps->prepend(s); delay = 50; }
		}
		else if (s == "ret") key(Qt::Key_Return, "\r");
		else if (s == "bs") key(Qt::Key_Backspace, "");
		else if (s == "exact") w->findChild<QPushButton *>("exact")->click();
		else if (s == "stop") stop->click();
		else if (s == "undo") QMetaObject::invokeMethod(board, "undo");
		else if (s.startsWith("sleep:")) delay = s.mid(6).toInt();
		else if (s == "hard") w->findChild<QCheckBox *>("hard")->toggle();
		else if (s == "pool") { auto * c = w->findChild<QComboBox *>("pool"); c->setCurrentIndex(1 - c->currentIndex()); }
		else if (s.startsWith("row:"))
		{
			auto * t = w->findChild<QTableView *>("table");
			t->setCurrentIndex(t->model()->index(s.mid(4).toInt(), 1));
		}
		else if (s.startsWith("tab:")) w->findChild<QTabWidget *>("tabs")->setCurrentIndex(s.mid(4).toInt());
		else if (s.startsWith("replay:")) w->replayGame(s.mid(7).toLong());
		else if (s == "next") w->findChild<QPushButton *>("replay_next")->click();
		else if (s == "prev") w->findChild<QPushButton *>("replay_prev")->click();
		else if (s == "games") QMetaObject::invokeMethod(w, "openGame", Qt::QueuedConnection);
		else if (s.startsWith("dlg:"))
		{
			auto * d = qobject_cast<QDialog *>(QApplication::activeModalWidget());
			if (!d) { steps->prepend(s); delay = 50; } // not open yet
			else if (s == "dlg:random") d->findChild<QPushButton *>("random")->click();
			else if (s.startsWith("dlg:tab:")) d->findChild<QTabWidget *>("tabs")->setCurrentIndex(s.mid(8).toInt());
			else if (s.startsWith("dlg:source:")) d->findChild<QComboBox *>("source")->setCurrentIndex(s.mid(11).toInt());
			else if (s.startsWith("dlg:row:")) d->findChild<QTableView *>("games")->selectRow(s.mid(8).toInt());
			else if (s == "dlg:open") d->accept();
			else if (s.startsWith("dlg:shot:")) d->grab().save(s.mid(9));
		}
		else if (s.startsWith("shot:")) w->grab().save(s.mid(5));
		else if (s == "quit") { QApplication::quit(); return; }
		else
			for (QChar c : s)
				key(c.isDigit() ? Qt::Key_0 + c.digitValue() : Qt::Key_A + c.toLower().unicode() - 'a', c);
		QTimer::singleShot(delay, [step] { (*step)(); });
	};
	QTimer::singleShot(200, *step);
}

int main(int argc, char ** argv)
{
	QApplication app(argc, argv);
	app.setApplicationName("wordle-gpu");
	app.setApplicationDisplayName("Wordle GPU Solver");
	try
	{
		MainWindow w(builtin_lists());
		w.show();
		const QString test = qEnvironmentVariable("WORDLE_GPU_TEST");
		if (!test.isEmpty()) run_test(&w, test.split(' ', Qt::SkipEmptyParts));
		return app.exec();
	}
	catch (const std::exception & e)
	{
		QMessageBox::critical(nullptr, "Wordle GPU Solver", QString("Could not start: %1").arg(e.what()));
		return 1;
	}
}
