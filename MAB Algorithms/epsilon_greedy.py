import pandas as pd
import numpy as np

def epsilon_greedy_algorithm(df, arms, epsilon=0.15, n_recommendations):
    
    #First, we have to chose if which arms we are going for, we will use a bernoulli :
    selector = np.random.binomial(1, epsilon)
    
    #With that, il selector == 1 (or for our initialisation == the dataframe is still empty), 
    #it means that we are going to chose random arms :
    if selector == 1 or df.shape[0]==0:
        recommendations = np.random.choice(a = arms, size = n_recommendations, replace = False)
        
    #Otherwise, we need to go for the arms with the highest empirical means :
    else:
        emp_mean = df.groupby('movie_id').agg({'binary_rating': ['mean']}) #We compute the empirical mean for each movie
        emp_mean = emp_mean['binary_rating'].sort_values(by = 'mean', ascending = False) #We sort the DataFrame
        if n_recommendations > 1:

            recommendations = emp_mean.iloc[:n_recommendations].index.tolist() #We get our movie_id with the highest mean

        else :

            recommendations = emp_mean.iloc[0]
        
    return recommendations

def policy_evaluator_epsilon(dataframe, epsilon_value = 0.15):
    # We get the list of the arms (the movies)
    arms = dataframe['movie_id'].unique().tolist()
    # We initialize an empty history
    history = pd.DataFrame([], columns = dataframe.columns)
    # We stock the rewards in a list
    rewards = []
    for t in range(len(dataframe)):
        # We check our t-th row
        t_event = dataframe[t : t + 1]
        # If the movie recommended matches the movie of our dataframe, we update our history and our reward
        if epsilon_greedy_algorithm(history, arms, epsilon=epsilon_value, n_recommendations=1) == t_event['movie_id'].iloc[0] :
            history.loc[len(history)] = t_event.iloc[0].to_list()
            rewards.append(t_event['binary_rating'].iloc[0])
    return rewards

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

def block_policy_evaluator_epsilon_2(dataframe, epsilon_value = 0.15, n_arms=5, block_size=50):
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
        recommendations = epsilon_greedy_algorithm(history, arms, epsilon=epsilon_value, n_recommendations=n_arms)
        # We focus on the matches
        good_recommendations = t_event[t_event['movie_id'].isin(recommendations)]
        # We update our history
        history_index.extend(good_recommendations.index.to_list())
        # We update the reward
        rewards.extend(good_recommendations['binary_rating'].to_list())
    return rewards