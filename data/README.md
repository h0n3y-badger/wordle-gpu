# Word lists

These files are data from other sources, not part of the Apache-2.0 licensed code:

- `allowed.txt`: Wordle's accepted guesses, from
  [tabatkins/wordle-list](https://github.com/tabatkins/wordle-list), MIT licensed
  (see `LICENSE-wordle-list.txt`).
- `answers.txt`: the original 2,315-word Wordle answer list, as published in
  [this gist](https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b).
- `nyt-answers.txt`: past New York Times Wordle answers by date, fetched from
  `nytimes.com/svc/wordle/v2/<date>.json` with `tools/wordle-gpu-update-answers`.

Played games and daily statistics are not here: `tools/wordle-gpu-import-games`
downloads them from their publishers into a local database (see the main README).

Wordle is a trademark of The New York Times Company; this project is not affiliated with it.
