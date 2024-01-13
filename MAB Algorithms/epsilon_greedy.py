import pandas as pd
import numpy as np
import random

def epsilon_greedy_algorithm_random(df, arms, n_recommendations=1, epsilon=0.15):
    # If several arms maximize the mean, we choose one randomly

    #First, we have to chose if which arms we are going for, we will use a bernoulli :
    selector = np.random.binomial(1, epsilon)
    
    #With that, il selector == 1 (or for our initialisation == the dataframe is still empty), 
    #it means that we are going to chose random arms :
    if selector == 1 or df.shape[0]==0:
        if n_recommendations == 1:
            recommendations = np.random.choice(a = arms, size = n_recommendations, replace = False)[0]
        else:
            recommendations = np.random.choice(a = arms, size = n_recommendations, replace = False).tolist()
        
    #Otherwise, we need to go for the arms with the highest empirical means :
    else:
        emp_mean = df.groupby('movie_id').agg({'binary_rating': ['mean']}) #We compute the empirical mean for each movie
        emp_mean.columns = ['mean']

        if n_recommendations > 1:
            #We get our movie_id's with the highest means
            recommendations = emp_mean.sort_values('mean', ascending=False).index.to_list()[:n_recommendations]

        else :
            # We check if there's only one argmax (one arm that maximizes the mean)
            max_value = emp_mean['mean'].max()
            if len(emp_mean[emp_mean['mean'] == max_value]) == 1:
                recommendations = emp_mean['mean'].idxmax()
            # If several arms are maximizing the mean, we pick one randomly
            else:
                list_of_arms = emp_mean[emp_mean['mean'] == emp_mean['mean'].max()].index.to_list()
                recommendations = random.choice(list_of_arms)
        
    return recommendations

def epsilon_greedy_algorithm_min(df, arms, n_recommendations=1, epsilon=0.15):
    # If several arms maximize the mean, we choose the movie with lowest ID.

    #First, we have to chose if which arms we are going for, we will use a bernoulli :
    selector = np.random.binomial(1, epsilon)
    
    #With that, il selector == 1 (or for our initialisation == the dataframe is still empty), 
    #it means that we are going to chose random arms :
    if selector == 1 or df.shape[0]==0:
        if n_recommendations == 1:
            recommendations = np.random.choice(a = arms, size = n_recommendations, replace = False)[0]
        else:
            recommendations = np.random.choice(a = arms, size = n_recommendations, replace = False).tolist()
        
    #Otherwise, we need to go for the arms with the highest empirical means :
    else:
        emp_mean = df.groupby('movie_id').agg({'binary_rating': ['mean']}) #We compute the empirical mean for each movie
        emp_mean.columns = ['mean']

        if n_recommendations > 1:
            #We get our movie_id's with the highest means
            recommendations = emp_mean.sort_values('mean', ascending=False).index.to_list()[:n_recommendations]

        else :
            # Even if several arms are maximizing the mean, we choose the arm with lowest id thanks to argmax
            recommendations = emp_mean['mean'].idxmax()
        
    return recommendations

def policy_evaluator_epsilon(dataframe, epsilon_value = 0.15):
    # We get the list of the arms (the movies)
    arms = dataframe['movie_id'].unique().tolist()
    # We initialize an empty history
    history = pd.DataFrame([], columns = dataframe.columns)
    # We stock the payoffs in a list
    payoffs = []
    for t in range(len(dataframe)):
        # We check our t-th row
        t_event = dataframe[t : t + 1]
        # If the movie recommended matches the movie of our dataframe, we update our history and our payoffs
        if epsilon_greedy_algorithm(history, arms, epsilon=epsilon_value, n_recommendations=1) == t_event['movie_id'].iloc[0] :
            history.loc[len(history)] = t_event.iloc[0].to_list()
            payoffs.append(t_event['binary_rating'].iloc[0])
    return payoffs

def block_policy_evaluator_epsilon(dataframe, epsilon_value = 0.15, block_size=50):
    # We get the list of the arms (the movies)
    arms = dataframe['movie_id'].unique().tolist()
    # We initialize an empty history
    history = pd.DataFrame([], columns = dataframe.columns)
    # We stock the rewards in a list
    rewards = []
    for t in range(len(dataframe) // block_size):
        # As explained, we consideer a block and not a single observation
        time = t*block_size
        t_event = dataframe[time : time + block_size]
        # If recommendations matches the movies in our data, We update our history and reward
        recommendations = epsilon_greedy_algorithm(history, arms, epsilon=epsilon_value, n_recommendations=block_size)
        # We focus on the matches
        good_recommendations = t_event[t_event['movie_id'].isin(recommendations)]
        # We update our history
        history = pd.concat([history, good_recommendations])
        # We update the reward
        rewards.extend(good_recommendations['binary_rating'].to_list())
    return rewards

def block_policy_evaluator_epsilon_faster(dataframe, epsilon_value = 0.15, block_size=50):
    # We get the list of the arms (the movies)
    arms = dataframe['movie_id'].unique().tolist()
    # We initialize an empty history
    history_index = []
    # We stock the rewards in a list
    rewards = []
    for t in range(len(dataframe) // block_size):
        # As explained, we consideer a block and not a single observation
        time = t*block_size
        t_event = dataframe[time : time + block_size]
        # If recommendations matches the movies in our data, We update our history and reward
        history = pd.DataFrame(dataframe.loc[history_index])
        recommendations = epsilon_greedy_algorithm(history, arms, epsilon=epsilon_value, n_recommendations=block_size)
        # We focus on the matches
        good_recommendations = t_event[t_event['movie_id'].isin(recommendations)]
        # We update our history
        history_index.extend(good_recommendations.index.to_list())
        # We update the reward
        rewards.extend(good_recommendations['binary_rating'].to_list())
    return rewards

# def epsilon_greedy_algorithm(df, arms, n_recommendations, epsilon=0.15):
#     #First, we have to chose if which arms we are going for, we will use a bernoulli :
#     selector = np.random.binomial(1, epsilon)
    
#     #With that, il selector == 1 (or for our initialisation == the dataframe is still empty), 
#     #it means that we are going to chose random arms :
#     if selector == 1 or df.shape[0]==0:
#         if n_recommendations == 1:
#             recommendations = np.random.choice(a = arms, size = n_recommendations, replace = False)[0]
#         else:
#             recommendations = np.random.choice(a = arms, size = n_recommendations, replace = False).tolist()
        
#     #Otherwise, we need to go for the arms with the highest empirical means :
#     else:
#         emp_mean = df.groupby('movie_id').agg({'binary_rating': ['mean']}) #We compute the empirical mean for each movie
#         emp_mean.columns = ['mean']
#         emp_mean.sort_values(by = 'mean', ascending = False, inplace=True) #We sort the DataFrame
#         if n_recommendations > 1:

#             recommendations = emp_mean.index.to_list()[:n_recommendations] #We get our movie_id with the highest mean

#         else :

#             recommendations = emp_mean.index[0]
        
#     return recommendations