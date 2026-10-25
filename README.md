# wordle-gpu

Finds the best next Wordle guess by brute force on an NVIDIA GPU, with a Qt GUI
and a command-line tool.

For every position it scores **every allowed guess** (14,855 words) against every
answer still possible:

1. **Split stats** (instant): colour-result groups, entropy, expected and
   worst-case answers left.
2. **Two-ply brute force** (GPU): every first guess × every colour result ×
   every second guess. From the opening that is 14,855 × 14,855 × 2,386 ≈ 5×10¹¹
   feedback lookups, about 0.5 s on an RTX 5070 Ti. It gives each guess the
   chance of finishing within two more guesses, and a lower bound on its average
   score (exact for two guesses, then the best any strategy could do).
3. **Exact solve** (CPU, branch and bound over all guesses at every node,
   pruned with the GPU's bounds): the true optimal average score, played
   perfectly from here within six guesses. Guesses that can't beat the best so
   far are refuted rather than solved, so with up to 3,000 possible answers it
   runs by itself and proves the best guess optimal. That takes about 0.9 s
   from the opening, 2.5 s in hard mode, and milliseconds later in the game.
   The "any valid word" opening (14,855 answers) is still too big to prove.

Rows marked "≥" are lower bounds: either nothing tighter was needed, or the
guess was shown to be worse than the best.

Exact opening results are kept in `~/.cache/wordle-gpu/` (one file per word
lists, answer list and hard mode); `wordle-gpu-cli -C` ignores the cache.

Opening results (2026-10-24 lists):

| Answers | Mode | Best opener | Average |
|---|---|---|---|
| original 2,315 | normal | TARSE | 3.4156 (7907 total) |
| original + NYT picks, 2,386 | normal | TARSE | 3.4233 (8168) |
| original + NYT picks, 2,386 | hard | LEAST (tied with SALET) | 3.5163 (8390) |

## Test hardware

All timings in this README were measured on:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (Blackwell, sm_120), driver 615.78 |
| CPU | AMD Ryzen 7 8700F, 8 cores / 16 threads (Zen 4, AVX-512) |
| RAM | 32 GB |
| Software | CUDA 13.4, GCC 16, Qt 6.12, Linux (Artix) |

The exact solver uses AVX-512 when built with `-march=native` on a CPU that has it,
and falls back to plain code otherwise. The GPU code needs an NVIDIA card; set
`CMAKE_CUDA_ARCHITECTURES` for cards other than sm_120.

## Build

Needs CUDA (nvcc, sm_120 by default), CMake, Ninja or Make, and Qt 6 Widgets
for the GUI (without Qt 6, only the CLI is built). SQLite 3 adds the played-games
database (`wordle-gpu-games` and the GUI's **Saved games…**).

    cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
    ninja -C build

Other GPUs: `-DCMAKE_CUDA_ARCHITECTURES=89` (etc.). Install to `~/.local`
(binaries plus a desktop entry):

    cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local && cmake --install build

## GUI

    ./build/wordle-gpu

Type a guess, click its tiles until they match the colours the game showed
(right-click goes back; keys 1-5 cycle a tile), press Enter. The table ranks every
guess. Double-click one to type it in. ● marks words that could still be the
answer. The panel below lists the possible answers and how the selected guess
would split them.

- **Hard mode** restricts guesses (and the lookahead's second guesses) to the
  ones the game allows.
- **Answers** switches between the original Wordle answer list and "any valid
  word", for when the answer may be outside the original list.
- **Solve exactly** runs the exact search until the best guess is proven
  optimal; **Stop** cancels it.
- **Saved games…** replays a game from the games database (see below) a guess at a
  time: each step shows the position before the guess that was played, and where
  that guess ranks against the best.

GPU work is cut into launches of about 25 ms so the desktop stays smooth.

## CLI

    ./build/wordle-gpu-cli tarse=bbbbb colin=ggbby
    ./build/wordle-gpu-cli -e 10 roate=bbbbb   # exact values for the best 10
    ./build/wordle-gpu-cli -p                  # prove the best opener (minutes)

Patterns are g/y/b (or 2/1/0) per letter. `-H` hard mode, `-a` any word can be
the answer, `-r` raw output for scripts, `-h` for the rest.

Time budget and interactive play:

    ./build/wordle-gpu-cli -T 30 tarse=bbbby     # spend up to 30 s, show the best found
    ./build/wordle-gpu-cli -i -T 2m              # play a game, up to 2 min per position

With `-T` the GPU brute force runs first, then the exact search keeps going until the best
guess is proven or the time is up. Ctrl-C stops early, and either way it prints the best ranking
so far. In `-i` mode type `tarse bbygb` per turn, plus `undo`, `new` and `quit`.

## Solving the whole game

    wordle-gpu-solve                 # until every opener is solved, or Ctrl-C
    wordle-gpu-solve -T 30m -k 50    # stop after 30 minutes, write games for the best 50
    wordle-gpu-solve -H tarse=bbbby  # hard mode, starting after a first guess

`wordle-gpu-solve` needs no game input. It runs the GPU brute force for every opening
guess, then solves openers exactly (no refutation shortcuts), most promising first,
until all are done, `-T` runs out or you press Ctrl-C. Then it writes:

- `PREFIX-openers.txt`: every opener ranked by average score (exact for the solved
  ones, a lower bound for the rest), with the guess-count spread and worst case for
  the best `-k`;
- `PREFIX-games.txt`: for the best `-k` openers (default 10, `-k all` for every solved
  one), every game the optimal strategy plays: the answer, the number of guesses and
  each guess with its colours, shortest games first.

PREFIX defaults to `wordle-solve-<answers>[-hard]` in the current directory (`-O` changes it).
Solving all 14,855 openers takes about an hour in normal mode (58 min on a Ryzen 7 8700F);
the best ones go fast, the weak ones leave big groups that are slow to solve exactly.

## Word lists

`data/allowed.txt` (14,855 accepted guesses), `data/answers.txt` (the original
2,315 answers) and `data/nyt-answers.txt` (every answer the NYT has used, by date) are
compiled in. The NYT picks answers by hand now; by 2026-10-24, 71 of them were outside
the original list (HOLEY, KEFIR, EMOJI…), and some answers have repeated. The default
answer pool is therefore the original list plus every NYT answer so far (2,386 words).
`-o` uses only the original list, and `-a` allows any valid word.

`wordle-gpu-update-answers` fetches newer NYT answers into
`~/.local/share/wordle-gpu/nyt-answers.txt`, which both programs read at startup. It
fetches only dates it doesn't have yet, and never later than yesterday unless you pass
`--through`, so it can't spoil today's puzzle. `-w` / `-A` load other lists.

## Played games

`wordle-gpu-import-games` downloads published Wordle games and daily statistics into
`~/.local/share/wordle-gpu/games.sqlite` (about 1.4 GB, plus 0.7 GB of downloads kept in
`~/.cache/wordle-gpu/downloads` for re-imports). Nothing of it is in this repository:
several sources are copyrighted or have no stated licence, so keep the database on your
own machine. `wordle-gpu-import-games -l` lists the sources; name some to import just those.

| Source | What | Games |
|---|---|---|
| `tweets-calvet` | Results shared on Twitter, Jan–Nov 2022 (Kaggle, CC BY-SA 4.0) | 6.9 M, colours only |
| `tweets-hamner` | Shared results, Wordle 210–519 (Kaggle) | 1.2 M, colours only, solved games only |
| `tweets-vora` | Shared results, Wordle 252–399 (Kaggle, CC0) | 2.0 M, colours only |
| `sim-hohmann` | Random play from the original answers (Kaggle, CC0) | 3.3 M with words |
| `lmrl-gym` | Scripted games from LMRL-Gym (Berkeley) | 1.1 M with words |
| `clembench-kto` | LLM players from clembench (Hugging Face) | 1,210 with words |
| `nyt-wordlebot` | WordleBot's daily guess counts, normal and hard mode, every day so far | daily |
| `mcm-2023` | The MCM 2023 contest's daily table for 2022 (percentages) | daily |

Tweets keep no names or text: each player gets an anonymous number per source. Every
game records whether its colours could be real for its answer. LMRL-Gym and clembench
colour repeated letters their own way (LMRL-Gym shows every extra copy yellow), so 34% and
10% of their games aren't positions the real game reaches; they are kept but marked, and
skipped when scoring.

    wordle-gpu-games                       # the sources
    wordle-gpu-games stats                 # solved, average and guess spread per source
    wordle-gpu-games day 2026-10-20        # one puzzle: WordleBot, MCM and shared results
    wordle-gpu-games -s lmrl-gym -R list   # random games (-w answer, -p puzzle, -W words, -c real colours)
    wordle-gpu-games -x show 13618400      # a game with each guess scored against the best
    wordle-gpu-games -n 500 analyze        # score 500 games per source, then a summary

Scoring replays the game: at each position it proves the best guess and solves the
played one exactly, and reports how many guesses the played one costs on average
compared with perfect play (`-q` uses the GPU's two-ply bounds instead). That takes about
half a second a game; results are kept in the database (`turn_eval`), so `show` and
later `analyze` runs reuse them.

## Checks

The stats, two-ply values (normal and hard mode) and exact values were checked
against an independent Python brute force on several positions, with no
differences.

## License

Apache License 2.0, see `LICENSE`. The word lists in `data/` come from other
sources; see `data/README.md`. Wordle is a trademark of The New York Times Company;
this project is not affiliated with it.
