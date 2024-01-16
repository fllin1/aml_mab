# aml_mab

## DataSet
https://grouplens.org/datasets/movielens/1m/

## Requirements

In order to have all the packages necessary to reproduce the code, make sure to execute the following command in your environment :

```bash
pip install -r requirements.txt
```

## Papers
- https://arxiv.org/pdf/1904.07272.pdf
- https://arxiv.org/pdf/1003.5956.pdf

## MAB Implementation
- Collection of Common Algorithms : https://arxiv.org/pdf/1402.6028.pdf
- Exp3 Algorithm : https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
- Neural Contextual Bandit w/ UCB-based Exploration : https://arxiv.org/pdf/1911.04462.pdf
- Stochastic MAB : https://proceedings.mlr.press/v48/degenne16.pdf

## Policy Evaluation

Our evaluation on each algorithm is based on a policy evaluation for finite data stream (https://arxiv.org/pdf/1003.5956.pdf). The context of policy_evaluator depends for each algorithm, thus we can't implement a function that will be the same for every algorithm, this is why there's a policy evaluator notebook for each algorithm.
We have slightly modified the policy_evaluator in the paper. We consideer payoffs as a list containing all the payoffs value instead of a single number to be able to plot the cumulative average of our payoffs, we then don't need to add the parameter T as it will be the len of payoffs.
We have implementend a policy evaluation that takes into account the fact that several arms can maximize the mean, thus we need to choose on arm. Two rules are possible and popular : choosing an arm randomly or choosing the arm with the lowest ID. There were no big differences between both algorithm, we have thus decided to keep the lowest ID rule.

It looks like with more data, our cumulative reward would still increase and so still hasn't converged totally.