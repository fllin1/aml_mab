import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def EXP_3_Tuned(t_event, w, eta, arms, N, rho, T):
    
    W_t = np.sum(w)
    Probas = []
    
    for a in range(N):
        P_a = (1-eta)*w[a]/W_t + eta/N
        Probas.append(P_a)
    A_t = np.random.choice(np.arange(N, dtype = int) , p = Probas)
    if arms[A_t] == t_event['movie_id'].iloc[0]:
        l_hat = (1 - t_event['rating'].iloc[0]) / Probas[A_t]
        w[A_t] = w[A_t]*np.exp( -eta * l_hat/N )
    else:
        l_hat = (rho*t_event['time'].iloc[0])/T / Probas[A_t]
        w[A_t] = w[A_t]*np.exp( -eta * l_hat/N )
    return (arms[A_t],w)

def policy_evaluator_EXP3_Tuned(dataframe, eta, rho):
    # We get the list of the arms (the movies)
    arms = dataframe['movie_id'].unique().tolist()
    N = len(arms)
    w = np.ones(N, dtype = float)
    # We stock the payoffs in a list
    payoffs = []
    T = len(dataframe)
    for t in range(T):
        # We check our t-th row
        t_event = dataframe[t : t + 1]
        a_t , w = EXP_3_Tuned(t_event , w, eta, arms, N, rho, T)
        # If the movie recommended matches the movie of our dataframe, we update our history and our payoffs
        if a_t == t_event['movie_id'].iloc[0] :
            payoffs.append(t_event['binary_rating'].iloc[0])
    return payoffs