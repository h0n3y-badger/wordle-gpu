# wordle-gpu

Finds the best next Wordle guess by brute force on an NVIDIA GPU, with a Qt GUI
and a command-line tool.

For every position it scores **every allowed guess** (14,855 words) against every
answer still possible:

1. **Split stats** (instant): colour-result groups, entropy, expected and
   worst-case answers left.
2. **Two-ply brute force** (GPU): every first guess × every colour result ×
   every second guess. From the opening that is 14,855 × 14,855 × 2,315 ≈ 5×10¹¹
   feedback lookups, about 1.2 s on an RTX 5070 Ti. It gives each guess the
   chance of finishing within two more guesses, and a lower bound on its average
   score (exact for two guesses, then the best any strategy could do).
3. **Exact solve** (CPU, branch and bound over all guesses at every node,
   pruned with the GPU's bounds): the true optimal average score, played
   perfectly from here within six guesses. Once about 400 answers or fewer
   remain it runs by itself and usually proves the best guess optimal in under
   a second. From the opening it takes a few minutes.

Opening result with these lists: **TARSE** is optimal at 3.4156 guesses on
average (7907 total over the 2315 answers). Proving it took 4 min 50 s.

## Build

Needs CUDA (nvcc, sm_120 by default), CMake, Ninja or Make, and Qt 6 Widgets
for the GUI (without Qt 6, only the CLI is built).

    cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
    ninja -C build

Other GPUs: `-DCMAKE_CUDA_ARCHITECTURES=89` (etc.).

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

GPU work is cut into launches of about 25 ms so the desktop stays smooth.

## CLI

    ./build/wordle-gpu-cli tarse=bbbbb colin=ggbby
    ./build/wordle-gpu-cli -e 10 roate=bbbbb   # exact values for the best 10
    ./build/wordle-gpu-cli -p                  # prove the best opener (minutes)

Patterns are g/y/b (or 2/1/0) per letter. `-H` hard mode, `-a` any word can be
the answer, `-r` raw output for scripts, `-h` for the rest.

## Word lists

`data/allowed.txt` (14,855 accepted guesses) and `data/answers.txt` (the
original 2,315 answers) are compiled in. `-w` / `-A` load others. The NYT now picks
answers by hand, and some recent ones are outside the original list. Switch
the answer list if a game rules everything out.

## Checks

The stats, two-ply values (normal and hard mode) and exact values were checked
against an independent Python brute force on several positions, with no
differences.
