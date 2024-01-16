import pandas as pd
import numpy as np


def EXP_3(t_event, w, eta, arms, N):
    
    W_t = np.sum(w)
    Probas = []
    
    for a in range(N):
        P_a = (1-eta)*w[a]/W_t + eta/N
        Probas.append(P_a)
    A_t = np.random.choice(np.arange(N, dtype = int) , p = Probas)
    
    if arms[A_t] == t_event['movie_id'].iloc[0]:
        l_hat = (1 - t_event['rating'].iloc[0]) / Probas[A_t]
        w[A_t] = w[A_t]*np.exp( -eta * l_hat/N )
        
    return (arms[A_t],w)


def policy_evaluator_EXP3(dataframe, eta, loops):
    print('exp3 begins')
    # We get the list of the arms (the movies)
    arms = dataframe['movie_id'].unique().tolist()
    N = len(arms)
    w = np.ones(N, dtype = float)
    # We stock the payoffs in a list
    payoffs = []
    
    for loop in range(loops):
        
        if loop == 0:
            # Start with filtered data first
            data = dataframe.copy()
            # Initiate unused_data list
            unused_data = []
        
        else:
            # Recycle unused data
            data = pd.DataFrame(unused_data, columns = dataframe.columns)
            # Initiate unused_data list
            unused_data = []
            
        print('Epoch ',loop )
        # We keep track of our progression
        decile = len(data)/10
        pourcent = 10
            
        for t in range(len(data)):
            
            if t > decile:
                print(f'progression : {pourcent} %')
                decile += len(data)/10
                pourcent += 10
            
            # We check our t-th row
            t_event = data[t : t+1]
            a_t , w = EXP_3(t_event , w, eta, arms, N)
            # If the movie recommended matches the movie of our dataframe, we update our history and our payoffs
            if a_t == t_event['movie_id'].iloc[0] :
                payoffs.append(t_event['binary_rating'].iloc[0])
            else:
                # Recycle data
                unused_data.append(data.iloc[t].tolist())
                
    return payoffs
