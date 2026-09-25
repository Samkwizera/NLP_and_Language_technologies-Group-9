# Vaccine tweet sentiment - Group 9

NLP & Language Technologies group project. We compare classical and sequential models on the Zindi "To Vaccinate or Not to Vaccinate" dataset (tweets labelled negative / neutral / positive towards vaccines, with annotator agreement).

| Member | Role |
|---|---|
| Samuel Kwizera Ihimbazwe | EDA, preprocessing, shared split, TF-IDF baselines, agreement experiments |
| Divine Okon Itu | BiLSTM / BiGRU, CNN / CNN-BiLSTM |
| Sheilla Keza Ruvugabigwi | Transformers: twitter-roberta / BERTweet vs XLM-R / AfroXLMR |

Models:
1. TF-IDF (word + char n-grams) + Logistic Regression
2. TF-IDF + Linear SVM / Naive Bayes
3. BiLSTM / BiGRU with GloVe Twitter or fastText
4. CNN / CNN-BiLSTM
5. Fine-tuned twitter-roberta / BERTweet vs XLM-R / AfroXLMR

Everyone uses the same stratified 70/15/15 split of Train.csv (seed 42, saved in `splits/`). The main metric is macro-F1. We also report accuracy, per-class F1, confusion matrices, learning curves and RMSE (to compare with the Zindi leaderboard). All runs are logged in `results/experiments.csv`.

## Running on Colab
1. In Colab go to File > Open notebook > GitHub and search `Samkwizera/NLP_and_Language_technologies-Group-9`.
2. Put `Train.csv` and `Test.csv` in a Google Drive folder called `nlp_data` (in MyDrive). You only need to do this once.
3. Run the first cell. It clones the repo, installs the requirements, mounts Drive and copies the csvs into `data/`.
4. Run all. Use a GPU runtime for notebooks 03 and 04.

## Structure
```
data/        Train.csv, Test.csv (not in git)
notebooks/   01_eda, 02_baselines, 03_rnn_cnn, 04_transformer
src/         config, preprocessing, split, dataset, models/, train, evaluate, utils
splits/      shared split ids
results/     figures/, errors/, experiments.csv
```
