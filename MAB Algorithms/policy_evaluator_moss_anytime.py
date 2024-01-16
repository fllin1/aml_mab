import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def policy_evaluator_moss_anytime_random(dataframe, alpha):
    # We stock the payoffs in a list
    payoffs = []
    # We pull each arm once to initialize history
    history = dataframe.groupby('movie_id').first()
    arms = dataframe['movie_id'].unique()
    n_arms = len(arms)
    history['movie_id'] = history.index
    # We drop the rows associated to the initial pull
    rows_to_drop = history['time']
    history['time'] = 0
    history.reset_index(drop=True, inplace=True)
    history = history[dataframe.columns]
    dataframe_copy = dataframe.copy()
    dataframe_copy.drop(rows_to_drop, inplace=True)
    dataframe_copy.reset_index(drop=True, inplace=True)
    dataframe_copy['time'] = dataframe_copy.index
    for t in range(1, len(dataframe_copy) + 1):
        # We get t-th row of our dataframe
        t_event = dataframe_copy[t-1:t]
        # We get the recommendation of our algorithm
        # The groupby allows to get s without adding a dictionary. It also allows us to not have a loop over each arm.
        objective_function = history[['movie_id', 'binary_rating']].groupby('movie_id').agg({'binary_rating': ['mean', 'count']})
        objective_function.columns = ['mean', 'count']
        objective_function['value'] = objective_function['mean'] + objective_function['count'].apply(lambda count: np.sqrt( ((1 + alpha) / 2) * max(np.log(t / (n_arms * count)), 0) / count)) 
        
        # We check if there's only one argmax (one arm that maximizes the objective_function)
        max_value = objective_function['value'].max()
        if len(objective_function[objective_function['value'] == max_value]) == 1:
            
            arm_chosen = objective_function['value'].idxmax() # Our movies id are the index of our dataframe
        # If several arms are maximizing the objective function, we pick one randomly
        else:
            list_of_arms = objective_function[objective_function['value'] == objective_function['value'].max()].index.to_list()
            arm_chosen = random.choice(list_of_arms)

        if arm_chosen == t_event['movie_id'].iloc[0]:
            history.loc[len(history)] = t_event.iloc[0].to_list()
            payoffs.append(t_event['binary_rating'].iloc[0])

    return payoffs

def policy_evaluator_moss_anytime_min(dataframe, alpha):
    # We stock the payoffs in a list
    payoffs = []
    # We pull each arm once to initialize history
    history = dataframe.groupby('movie_id').first()
    arms = dataframe['movie_id'].unique()
    n_arms = len(arms)
    history['movie_id'] = history.index
    # We drop the rows associated to the initial pull
    rows_to_drop = history['time']
    history['time'] = 0
    history.reset_index(drop=True, inplace=True)
    history = history[dataframe.columns]
    dataframe_copy = dataframe.copy()
    dataframe_copy.drop(rows_to_drop, inplace=True)
    dataframe_copy.reset_index(drop=True, inplace=True)
    dataframe_copy['time'] = dataframe_copy.index
    for t in range(1, len(dataframe_copy) + 1):
        # We get t-th row of our dataframe
        t_event = dataframe_copy[t-1:t]
        # We get the recommendation of our algorithm
        # The groupby allows to get s without adding a dictionary. It also allows us to not have a loop over each arm.
        objective_function = history[['movie_id', 'binary_rating']].groupby('movie_id').agg({'binary_rating': ['mean', 'count']})
        objective_function.columns = ['mean', 'count']
        objective_function['value'] = objective_function['mean'] + objective_function['count'].apply(lambda count: np.sqrt( ((1 + alpha) / 2) * max(np.log(t / (n_arms * count)), 0) / count)) 
        
        # Even if several arms are maximizing the mean, we choose the arm with lowest id thanks to argmax
        arm_chosen = objective_function['value'].idxmax() # Our movies id are the index of our dataframe

        if arm_chosen == t_event['movie_id'].iloc[0]:
            history.loc[len(history)] = t_event.iloc[0].to_list()
            payoffs.append(t_event['binary_rating'].iloc[0])

    return payoffs

def policy_evaluator_moss_anytime_slower(dataframe, alpha):
    # First version of policy_evaluator_min. It's much slower but still works...
    # We stock the payoffs in a list
    payoffs = []
    # We pull each arm once to initialize history
    history = dataframe.groupby('movie_id').first()
    arms = dataframe['movie_id'].unique()
    n_arms = len(arms)
    history['movie_id'] = history.index
    # We drop the rows associated to the initial pull
    rows_to_drop = history['time']
    history['time'] = 0
    dataframe_copy = dataframe.copy()
    dataframe_copy.drop(rows_to_drop, inplace=True)
    dataframe_copy.reset_index(drop=True, inplace=True)
    dataframe_copy['time'] = dataframe_copy.index
    # We create a dict containing all the movies ID and their corresponding number of pulls
    s = {}
    for movie in arms:
        s[movie] = 1
    for t in range(1, len(dataframe_copy) + 1):
        # We get t-th row of our dataframe
        t_event = dataframe_copy[t-1:t]
        # We get the recommendation of our algorithm
        objective_function = {}
        for k in arms:
            objective_function[k] = history[history['movie_id'] == k]['binary_rating'].mean() + np.sqrt( ( (1 + alpha) / 2) * max(np.log( t / (n_arms * s[k] ) ), 0) / s[k] )
        
        arm_chosen = max(objective_function, key=objective_function.get)

        s[arm_chosen] += 1

        if arm_chosen == t_event['movie_id'].iloc[0]:
            history.loc[len(history)] = t_event.iloc[0].to_list()
            payoffs.append(t_event['binary_rating'].iloc[0])

    return payoffs