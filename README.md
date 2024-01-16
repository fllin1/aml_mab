# aml_mab

## DataSet
https://grouplens.org/datasets/movielens/1m/

## Papers
- Introduction to MAB : https://arxiv.org/pdf/1904.07272.pdf
- Policy Evaluator : https://arxiv.org/pdf/1003.5956.pdf

## MAB Implementation
- Collection of Common Algorithms : https://arxiv.org/pdf/1402.6028.pdf
- Exp3 Algorithm : https://inst.eecs.berkeley.edu/~ee290s/fa18/scribe_notes/EE290S_Lecture_Note_22.pdf
- MOSS Algorithm : https://www.di.ens.fr/willow/pdfscurrent/COLT09a.pdf
- MOSS Anytime Algorithm : https://proceedings.mlr.press/v48/degenne16.pdf
- LinUCB Algorithm : https://arxiv.org/pdf/1003.0146.pdf
- Neural Contextual Bandit w/ UCB-based Exploration : https://arxiv.org/pdf/1911.04462.pdf

## How to recreate the plots for the big DataSet (6 millions observations after processing)
This section concerns the plots for the Epsilon Greedy, MOSS Anytime and EXP3 algorithms.
Simply open the notebook "plots_for_20M.ipynb" in the file "Plot Big Data" and run all cells, it will plot the graphs and their respective csv in the same file. You can chose the number of epochs by changing the parameter "loops" for the MOSS Anytime and EXP3 algorithms.
Our plots can be found in the "Plot Big Data/results" file.
