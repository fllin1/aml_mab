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