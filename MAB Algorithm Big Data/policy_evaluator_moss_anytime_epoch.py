import pandas as pd
import numpy as np

def policy_evaluator_moss_anytime_min(dataframe, alpha, epochs):
    print('moss anytime begins')
    
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

    for epoch in range(epochs):
        
        if epoch == 0:
            # Initiate unused_data list
            used_data_index = []
        
        else:
            # Recycle unused data
            dataframe_copy.drop(used_data_index, inplace = True)
            # Initiate unused_data list
            used_data_index = []
            
        print('Epoch ',epoch+1 )
        # We initialize our quantities to keep track of our progression
        decile = len(dataframe_copy)/10
        pourcent = 10
            
        for t in range(1, len(dataframe_copy) + 1):
            
            # We keep track of our progression
            if t > decile:
                print(f'progression : {pourcent} %')
                decile += len(dataframe_copy)/10
                pourcent += 10
            
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
                used_data_index.append(t)

    return payoffs