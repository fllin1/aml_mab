# Around the Multi-Armed Bandit Problem

## DataSet
https://grouplens.org/datasets/movielens/1m/
https://grouplens.org/datasets/movielens/25m/

## Requirements

In order to have all the packages necessary to reproduce the code, make sure to execute the following command in your environment :

```bash
pip install -r requirements.txt
```

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

## Policy Evaluation

Our evaluation on each algorithm is based on a policy evaluation for finite data stream (https://arxiv.org/pdf/1003.5956.pdf). The context of policy_evaluator depends for each algorithm, thus we can't implement a function that will be the same for every algorithm, this is why there's a policy evaluator notebook for each algorithm.
We have slightly modified the policy_evaluator in the paper. We consideer payoffs as a list containing all the payoffs value instead of a single number to be able to plot the cumulative average of our payoffs, we then don't need to add the parameter T as it will be the len of payoffs.
We have implementend a policy evaluation that takes into account the fact that several arms can maximize the mean, thus we need to choose on arm. Two rules are possible and popular : choosing an arm randomly or choosing the arm with the lowest ID. There were no big differences between both algorithm, we have thus decided to keep the lowest ID rule.

## What contains this repository

Everytime a file contains the mention "big data", it indicates that it was intended for our bigger data base, and it was adapted from their respective file that doesn't contain the mention.

* Data : It contains the raw data (.dat files) and the processed dataset (dataset_1000_minimum_reviews.csv).
  * respectively Big Data : It contains the READ.ME to download the bigger dataset and what to do with it.
* MAB Algorithms : It contains the code about each algorithm and the policy evaluator for each of them. It also allows us to keep in memory the list of algorithms rewards.
  * respectively MAB Algorithms Big Data : There are only slight differences between these and the previous files, like the mention "_epoch" when epoches were added
* Plots : It contains the datasets and plots (in the Images folder) about all the cumulative evaluations of each algorithm and some describing charts related to the dataset. The plots in the Images folder are obtained by running data plot.ipynb and algorithms_plots.ipynb.
  * respectively Plots Big Data : the plots were realized only for what seemed like the optimal parameters
* Processing : It contains the notebook about how we processed both out datas in order to obtain our working dataset (dataset_1000_minimum_reviews.csv and big_dataset_20000_minimum_reviews.csv).

## How to reproduce the code

The following steps are similar if you want to work on either the big or small datasets.

We remind that you should first download yourself the big dataset and place the "rating.csv" in the "Big Data" file before working with the bigger dataset.

* Generate the processed dataset : Go to the Processing folder and open the processing notebook. Just "Run All" or follow the instructions in the notebook, all the processing is explained. The raw data and the processed dataset (dataset_1000_minimum_reviews.csv/ big_dataset_20000_minimum_reviews.csv) are in the Data (resp. Big Data) folder.
* Get the list of cumulative evaluations of each algorithm : Run the different simulations notebook in the MAB Algorithms (resp. MAB Algorithms Big Data) folder. The list of cumulative evaluations are stored in the Plots\Data (resp. Plots Big Data\Data) folder.
* Get all the different plots : Run the data plot.ipynb and algorithms_plots.ipynb in the Plots folder.
