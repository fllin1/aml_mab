import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def policy_evaluator_disjoint_lin_ucb(dataframe, alpha):
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
    # We get the number of user features
    d = len(dataframe.iloc[0][4:])
    # We initialize the different quantities used in our LinUCB
    probability = np.zeros(n_arms)

    A = [np.eye(d)] * n_arms
    A_inverse = A.copy()
    b = [np.zeros(d)] * n_arms
    theta = [np.matmul(A[0], b[0])] * n_arms
    
    for t in range(1, len(dataframe_copy) + 1):
        # We get t-th row of our dataframe
        t_event = dataframe_copy[t-1:t]
        # We get the recommendation of our algorithm
        features = np.array(t_event.iloc[0][4:])
        for arm in range(n_arms):

            # We compute probability for each arm
            probability[arm] = np.matmul( theta[arm], features ) + alpha * np.sqrt(np.matmul( np.matmul(features.T, A_inverse[arm]), features))
        
        # Even if several arms are maximizing the mean, we choose the arm with lowest id thanks to argmax
        index_arm_chosen = probability.argmax()
        arm_chosen = arms[index_arm_chosen]

        # If arm is chosen, we update history, rewards, but also update the quantities depending on our arm
        if arm_chosen == t_event['movie_id'].iloc[0]:
            history.loc[len(history)] = t_event.iloc[0].to_list()
            payoffs.append(t_event['binary_rating'].iloc[0])

            features_vector = features.copy()
            features_vector.shape = (d, 1)
            A[index_arm_chosen] = A[index_arm_chosen] + np.matmul(features_vector, features_vector.T)
            A_inverse[index_arm_chosen] = np.linalg.inv(A[index_arm_chosen])
            b[index_arm_chosen] = b[index_arm_chosen] + t_event['binary_rating'].iloc[0] * features
            theta[index_arm_chosen] = np.matmul(A_inverse[arm], b[arm])
            
    return payoffs