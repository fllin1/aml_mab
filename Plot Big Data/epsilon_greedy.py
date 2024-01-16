import pandas as pd
import numpy as np

def epsilon_greedy_algorithm_min(df, arms, n_recommendations=1, epsilon=0.15):
    # If several arms maximize the mean, we choose the movie with lowest ID.

    #First, we have to chose if which arms we are going for, we will use a bernoulli :
    selector = np.random.binomial(1, epsilon)
    
    #With that, il selector == 1 (or for our initialisation == the dataframe is still empty), 
    #it means that we are going to chose random arms :
    if selector == 1 or df.shape[0]==0:
        recommendations = np.random.choice(a = arms, size = n_recommendations, replace = False)[0]
        
    #Otherwise, we need to go for the arms with the highest empirical means :
    else:
        emp_mean = df.groupby('movie_id').agg({'binary_rating': ['mean']}) #We compute the empirical mean for each movie
        emp_mean.columns = ['mean']
        
        # Even if several arms are maximizing the mean, we choose the arm with lowest id thanks to argmax
        recommendations = emp_mean['mean'].idxmax()
        
    return recommendations

def policy_evaluator_epsilon(dataframe, epsilon_value = 0.15):
    print('epsilon greedy begins')
    # We get the list of the arms (the movies)
    arms = dataframe['movie_id'].unique().tolist()
    # We initialize an empty history
    history = pd.DataFrame([], columns = dataframe.columns)
    # We stock the payoffs in a list
    payoffs = []
    decile = len(dataframe)/10
    pourcent = 10
    for t in range(len(dataframe)):
        if t > decile:
            print(f'progression : {pourcent} %')
            decile += len(dataframe)/10
            pourcent += 10
        # We check our t-th row
        t_event = dataframe[t : t + 1]
        # If the movie recommended matches the movie of our dataframe, we update our history and our payoffs
        if epsilon_greedy_algorithm_min(history, arms, epsilon=epsilon_value, n_recommendations=1) == t_event['movie_id'].iloc[0] :
            history.loc[len(history)] = t_event.iloc[0].to_list()
            payoffs.append(t_event['binary_rating'].iloc[0])
    return payoffs