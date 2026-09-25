# Data

The csv files aren't in the repo. Download them from the Zindi competition "To Vaccinate or Not to Vaccinate" and save them here as `Train.csv` and `Test.csv` (the downloads might be called `train (1).csv` / `test (1).csv`).

Columns: `tweet_id`, `safe_text`, `label` (-1.0 / 0.0 / 1.0), `agreement` (1.0, 0.667 or 0.333). Test only has the first two.

Known issues, both handled in `src/preprocessing.py`:
- In Train, tweet `RQMQ0L2A` has a newline in it, so it's split over two lines with the columns shifted. We merge it back (label 1, agreement 0.667), which gives 10,000 rows.
- In Test, there's one line "Dr. JAMES SHANNON" with no ID (probably `H0VUUY2P`). We keep it with an empty id.
