# Sentiment benchmark

Datasets: smsa (n=1260), nusax (n=400), nusax_clean (n=81). Subset = the same 200 random items per dataset that Gemini saw (fair comparison).
`*` = contaminated: the model was fine-tuned on SmSA, and NusaX's Indonesian text comes from the same corpus.
`nusax_clean` = NusaX test items with no 8-gram overlap with SmSA train (fair to all models).

| model | context | dataset | acc | macro-F1 | F1 neg / neu / pos | subset acc | subset macro-F1 | ms/item |
|---|---|---|---|---|---|---|---|---|
| indonesia-bert-sentiment-classification* | none | nusax | 0.960 | 0.955 | 0.98 / 0.92 / 0.96 | 0.955 | 0.950 | 112.7 |
| indonesian-roberta-base-sentiment-classifier* | none | nusax | 0.907 | 0.894 | 0.94 / 0.82 / 0.92 | 0.890 | 0.873 | 121.9 |
| indobert-sentiment-classifier | review | nusax | 0.630 | 0.613 | 0.87 / 0.54 / 0.43 | 0.640 | 0.621 | 388.3 |
| indobert-sentiment-classifier | generic | nusax | 0.540 | 0.461 | 0.88 / 0.49 / 0.01 | 0.545 | 0.463 | 365.5 |
| indonesia-bert-sentiment-classification | none | nusax_clean | 0.938 | 0.929 | 1.00 / 0.90 / 0.89 | 0.938 | 0.929 | 72.5 |
| indonesian-roberta-base-sentiment-classifier | none | nusax_clean | 0.864 | 0.851 | 0.93 / 0.77 / 0.85 | 0.864 | 0.851 | 72.9 |
| indobert-sentiment-classifier | review | nusax_clean | 0.679 | 0.639 | 0.81 / 0.65 / 0.46 | 0.679 | 0.639 | 303.1 |
| indobert-sentiment-classifier | generic | nusax_clean | 0.617 | 0.503 | 0.81 / 0.61 / 0.10 | 0.617 | 0.503 | 344.1 |
| indonesia-bert-sentiment-classification* | none | smsa | 0.938 | 0.912 | 0.92 / 0.85 / 0.96 | 0.935 | 0.920 | 134.1 |
| indonesian-roberta-base-sentiment-classifier* | none | smsa | 0.925 | 0.892 | 0.91 / 0.81 / 0.95 | 0.930 | 0.906 | 121.7 |
| indobert-sentiment-classifier | review | smsa | 0.450 | 0.469 | 0.80 / 0.25 / 0.35 | 0.520 | 0.512 | 501.4 |
| indobert-sentiment-classifier | generic | smsa | 0.371 | 0.385 | 0.86 / 0.22 / 0.07 | 0.450 | 0.418 | 419.3 |

Confusion matrices (rows = gold neg/neu/pos, cols = predicted) and worst errors: `results.json`.
