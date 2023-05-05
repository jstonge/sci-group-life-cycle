import heapq
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgi
from numpy.random import binomial, choice, exponential, rand

from helpers import count_prog_nonprog, grab, plot_frac_prog_group, plot_pref

sns.set_style("whitegrid")


# helpers -----------------------------------------------------------------------------------------


# def get_max_prog_or_nonprog(history_group):
#     print(np.max([np.max(_) for _ in np.argwhere(history_group[-1] > 0)]))

def count_prog_nonprog(H, group):
    """return (p, n)"""
    # assert H.edges.size.asdict()[group] > 0, f"Group {group} has been passed to count_prog_nonprog but it is empty"
    if H.edges.size.asdict()[group] == 0:
       return 0, 0
    else:
        states_count = Counter(H.nodes[n]['state'] for n in H.edges.members(group))
        if len(states_count) == 2:
            return states_count.values()
        elif states_count.get('prog') is None:
            return 0, list(states_count.values())[0]
        else:
            return list(states_count.values())[0], 0

def τ(n, p, α, β, b=0.9): # group benefits
    return np.exp(-α + β*(1 - c(n, p, b=b)))

def c(n, p, a=3, b=0.9): # cost function
    return b if n == p == 0 else b * np.exp(-a*p / n) 

def plot_group_dist(H):
    count_members = Counter([len(e) for e in H.edges.members()])
    df = pd.DataFrame({"n": count_members.values(), "k": count_members.keys()})
    sns.barplot(x="k", y="n", data=df, color="darkblue", alpha=0.5)
    plt.xlabel("group size")
    plt.ylabel("count")

def init_hypergraph(nb_groups=1000, max_group_size=40):
    current_node = 0
    hyperedge_dict = {}
    for group in range(nb_groups):
        group_size = binomial(max_group_size, 0.1)
        
        while group_size == 0: # make sure group size > 0
            group_size = binomial(max_group_size, 0.1)
        
        group_members = []
        for node in range(current_node, current_node+group_size):
            group_members.append(node)
        hyperedge_dict.update({group: group_members})
        current_node += group_size

    return xgi.Hypergraph(hyperedge_dict)


def whats_happening(event_queue, H, group, params, t):
    mu, nu_n, nu_p, alpha, beta, b = params
    p, n = count_prog_nonprog(H, group)

    R_nonprog_grad = nu_n * n
    R_prog_grad = nu_p * p
    R_new_nonprog = mu * (1 + (n+p) * 0)
    R_conversion_attempt = τ(n, p, alpha, beta, b=b) if n > 0 else 0.
    
    # draw time to next event and event type
    R = R_nonprog_grad + R_conversion_attempt + R_new_nonprog + R_prog_grad
    tau = exponential(scale=1/R)
    
    taulab = ["non-prog graduates", "programmer graduates", "conversion attempt", "new non-programmer"]
    tauR = [R_nonprog_grad/R, R_prog_grad/R, R_conversion_attempt/R, R_new_nonprog/R] 
    which_event = choice(taulab, p=tauR)
    
    next_event = (tau, group, which_event, p, n)
    
    if which_event == 'conversion attempt':
        if R_conversion_attempt/R < (1-c(n,p,b=b)):
            next_event = (tau, group, "new programmer", p, n)
        else:
            next_event = (tau, group, "non-prog leaves", p, n)


    heapq.heappush(event_queue, next_event)

    return event_queue

def main():
    #parameters
    mu  = 0.1
    nu_n = 0.01
    nu_p = 0.05
    alpha  = 0.01
    beta = 0.1
    # a = 3.   
    b = .5
    params = (mu, nu_n, nu_p, alpha, beta, b)

    #initial conditions
    event_queue = []
    max_group_size=40
    H = init_hypergraph(max_group_size=max_group_size)
    I0 = 0.1
    I = 0
    Ig = np.zeros((max_group_size+1,max_group_size+1))
    time = 0
    # we start out we a hypergraph where hyperedges == groups
    # they are non-overlapping at the moment. all_gsize > 0.
    for group, members in enumerate(H.edges.members()): #loop over groups
        # group, members = 0, H.edges.members(0)
        gsize = len(members)
        states_binary = binomial(gsize, I0, size=gsize)
        nb_prog = np.sum(states_binary)
        states = np.where(states_binary == 0, 'non-prog', 'prog')
        for node, state in zip(members, states):
            H.nodes[node]['state'] = state
        event_queue = whats_happening(event_queue, H, group, params, time)
        Ig[nb_prog, gsize-nb_prog] += 1
        I += nb_prog

    # what's in the event queue?
    # Counter(e[2] for e in event_queue)

    history = []
    history_group = []
    tot_pop = []
    history.append(I/H.num_nodes)
    history_group.append(Ig/np.sum(Ig))
    tot_pop.append(H.num_nodes)
    times = np.zeros(1)

    #for each generation
    t_max = 5000
    while time < t_max and len(event_queue) > 0:
        # draw from event queue
        (tau, group, event, p, n) = heapq.heappop(event_queue)

        # we make sure that this group is not None.
        # this might happen when groups get depleted as people leave.
        # this is alright.
        while H.edges.get(group) is None:
            (tau, group, event, p, n) = heapq.heappop(event_queue)

        # in any case, something we'll happen and current group size will change.
        # This state should always > 0 as we are currently occupying it.

        [H.nodes[n] for n in H.edges.members(group)]

        if event == 'non-programmer graduates' or event == "non-prog leaves":           
            leaving_node = grab(H, group, 'non-prog')
            # Not ideal.
            assert leaving_node is not None, "node is None"
            if leaving_node:
                H.remove_node(leaving_node)
                assert n > 0
                Ig[p, n]  -= 1
                Ig[p, n-1] += 1
                
        if event == 'programmer graduates':
            leaving_node = grab(H, group, 'prog')
            assert leaving_node is not None, "node is None"
            # Not ideal.
            if leaving_node:
                H.remove_node(leaving_node)
                I -= 1
                assert p > 0
                Ig[p, n]  -= 1
                Ig[p-1, n] += 1
    
        if event == 'new non-programmer':
            new_node = H.num_nodes+1
            H.add_node_to_edge(group, new_node)
            H.nodes[new_node]['state'] = 'non-prog'
            assert n <= max_group_size
            Ig[p, n]  -= 1
            Ig[p, n+1] += 1
        
        if event == 'new programmer':
            converting_node = grab(H, group, 'non-prog')
            assert converting_node is not None, "node is None"
            # Not ideal.
            if converting_node:
                H.nodes[converting_node]['state'] = 'prog'
                Ig[p, n]  -= 1
                I += 1
                assert p <= max_group_size and n > 0
                Ig[p+1, n-1] += 1
                # Now that we have a new programmer in that group, something else might happen.
                # Another students might join the university, another student in the same group
                # might try it too, etc.

        if np.sum(Ig/np.sum(Ig) < 0):
            print(f"we got a negative value in IG at step {time}\nthe event was {event}\nGroup size was {len(H.edges.members(group))}\nn,p = {n, p}")
            break

        event_queue = whats_happening(event_queue, H, group, params, time)

        #update history
        time = time + tau
        tot_pop.append(H.num_nodes)
        history = np.append(history, I / H.num_nodes)
        history_group.append(Ig/np.sum(Ig))
        times = np.append(times, time)


    plot_frac_prog_group(history_group, times)


    #print time series
    plt.plot(times, history, marker="o", ls='--', color='black', alpha=0.05)
    plt.ylabel('Frac Programmers')
    plt.legend()
    plt.xlabel('Time')
    plt.plot(times[-1],history[-1], marker="o", ls='--', color='black', alpha=0.05, label='Simulations')
    plt.show()

    plt.plot(times, tot_pop, color='midnightblue', alpha=0.5)
    plt.ylabel('Total Population') # pop decay for some reason
    plt.legend()
    plt.xlabel('Time')
    plt.show()

    # plot_pref(history_pref)


main()


  





# old ------------------------------------------------------------------------

# def whats_happening(event_queue, H, group, mu, nu_n, nu_p, alpha, beta, t):
#     p, n = count_prog_nonprog(H, group)

#     tau_nonprog_grad = exponential(scale = 1 / (nu_n * n) ) if n != 0  else 999
#     tau_prog_grad = exponential(scale = 1 / (nu_p * p) ) if p != 0  else 999
#     tau_new_nonprog = exponential(scale = 1 / mu )
#     tau_conversion_attempt = exponential(scale = 1 / ( τ(n, p, alpha, beta) ) ) if n > 0 else 999

#     taus = [tau_nonprog_grad, tau_prog_grad, tau_new_nonprog, tau_conversion_attempt]
#     taulab = ['non-programmer graduates', 'programmer graduates', 'new non-programmer', 'conversion attempt']
#     min_tau = np.argmin(taus)

#     next_event = (t+taus[min_tau], group, taulab[min_tau])

#     if taulab[min_tau] == 'conversion attempt':
#         if tau_conversion_attempt > c(n,p): # tau > c --> fails 
#             tau_conversion_fails = exponential(scale=1 / ( τ(n, p, alpha, beta) * c(n, p)) )
#             next_event = (t + tau_conversion_fails, group, "non-programmer leaves")
#         else: # tau > (1-c) --> succeed
#             tau_conversion_outcome = exponential(scale=1 / ( τ(n, p, alpha, beta) * (1 - c(n, p)) ) )
#             next_event = (t + tau_conversion_outcome, group, "new programmer")

#     heapq.heappush(event_queue, next_event)

#     return event_queue


# Create a network
# H = init_hypergraph()

# # print(f"The hypergraph has {H.num_nodes} nodes and {H.num_edges} edges")
# # plot_group_dist(H)

# # Event queue for continuous time events
# event_queue = []

# # Parameters of the model
# mu  = 0.5
# nu_n = 0.01
# nu_p = 0.05
# alpha  = 0.01
# beta = 0.1 
# a = 3.
# I0 = 0.1 #initial fraction of programmers

# # Initial conditions
# t = 0
# I = 0

# for group, members in enumerate(H.edges.members()): #loop over groups
#     gsize = len(members)
#     states_binary = binomial(gsize, I0, size=gsize)
#     states = np.where(states_binary == 0, 'non-prog', 'prog')
#     for node, state in zip(members, states):
#         H.nodes[node]['state'] = state
#     event_queue = whats_happening(event_queue, H, group, mu, nu_n, nu_p, alpha, beta, t)
#     I += sum(states_binary)
        
# # Counter
# history = []
# times = []
# history.append(I/H.num_nodes)
# times.append(t)


# # Simulation
# tmax = 500
# while t<tmax and len(event_queue) > 0:
#     (time, group, event) = heapq.heappop(event_queue)
#     if event == 'non-programmer graduates' or event == "non-programmer leaves":
#         H.remove_node(grab(H, group, 'non-prog'))

#     if event == 'programmer graduates':
#         H.remove_node(grab(H, group, 'prog'))
#         I -= 1
    
#     if event == 'new non-programmer':
#         all_gsizes = H.edges.size().asnumpy()
#         dist_gsize = all_gsizes/all_gsizes.sum(axis=0,keepdims=1)
#         new_node = H.num_nodes+1
#         group = choice(H.edges, p=dist_gsize)
#         H.add_node_to_edge(group, new_node)
#         H.nodes[new_node]['state'] = 'non-prog'

#     if event == 'new programmer':
#         H.nodes[grab(H, group, 'non-prog')]['state'] = 'prog'
#         event_queue = whats_happening(event_queue, H, group, mu, nu_n, nu_p, alpha, beta, t)
#         I += 1
    
#     t=time
#     history.append(I/H.num_nodes)
#     times.append(t)


# plt.plot(times,history, marker="o", ls='--', color='black', alpha=0.25, label='Simulations')
# plt.legend()
# plt.ylabel('frac programmers')
# plt.xlabel('Time')
# plt.show()

