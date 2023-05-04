import heapq
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgi
from numpy.random import binomial, choice, exponential, rand

sns.set_style("whitegrid")

# helpers -----------------------------------------------------------------------------------------


def count_prog_nonprog(H, group):
    """return (p, n)"""
    states_count = Counter(H.nodes[n]['state'] for n in H.edges.members(group))
    if len(states_count) == 2:
        return states_count.values()
    elif states_count.get('prog') is None:
        return 0, list(states_count.values())[0]
    else:
        return list(states_count.values())[0], 0

def plot_group_dist(H):
    count_members = Counter([len(e) for e in H.edges.members()])
    df = pd.DataFrame({"n": count_members.values(), "k": count_members.keys()})
    sns.barplot(x="k", y="n", data=df, color="darkblue", alpha=0.5)
    plt.xlabel("group size")
    plt.ylabel("count")

def init_hypergraph():
    nb_groups = 1000
    max_group_size = 40
    
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
 
def τ(n, p, α, β, b=0.9): # group benefits
    return np.exp(-α + β*(1 - c(n, p, b=b)))

def c(n, p, a=3, b=0.9): # cost function
    return b if n == p == 0 else b * np.exp(-a*p / n) 

def grab(H, group, state):
    for n in H.edges.members(group):
        if H.nodes[n]['state'] == state:
            return n        

def plot_pref(history_pref):
    count_df = pd.DataFrame(Counter(history_pref).most_common(50), columns=['group', '# selected'])\
                 .assign(group = lambda x: x.group.astype(str))
    
    fig, ax = plt.subplots(1,1,figsize=(5, 10))
    sns.barplot(y="group", x="# selected", data=count_df, ax=ax)

def plot_frac_prog_group(history_group, times, out=None):
    history_group_wrangled = np.zeros((len(times), 40//2))
    for t in range(len(times)):
        out = np.zeros(40//2)
        for gsize in range(40//2):
            for p in range(gsize):
                n = gsize-p
                out[gsize] += history_group[t][p, n]
    
        history_group_wrangled[t,:] = out        

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.plot(times, [history_group_wrangled[t][1] for t in range(len(times))], 
            marker="o", ls='--', color='red', alpha=0.15, label="gsize=1")
    ax.plot(times, [history_group_wrangled[t][3] for t in range(len(times))], 
            marker="o", ls='--', color='black', alpha=0.15, label="gsize=3")
    ax.plot(times, [history_group_wrangled[t][5] for t in range(len(times))], 
            marker="o", ls='--', color='blue', alpha=0.15, label="gsize=5")
    ax.plot(times, [history_group_wrangled[t][7] for t in range(len(times))], 
            marker="o", ls='--', color='orange', alpha=0.15, label="gsize=7")
    ax.plot(times, [history_group_wrangled[t][10] for t in range(len(times))], 
            marker="o", ls='--', color='green', alpha=0.15, label="gsize=10")
    ax.plot(times, [history_group_wrangled[t][12] for t in range(len(times))], 
            marker="o", ls='--', color='cyan', alpha=0.15, label="gsize=12")
    plt.legend()
    plt.ylabel("frac programmers")
    plt.xlabel("Time")

    if out is not None:
        plt.savefig("test_sims.png")


# Hybrid next-reaction-per-group method ------------------------------------------------------------

def pref_attachment_new_node(H):
    """return which new group new node we'll attach to"""
    all_gsizes = H.edges.size().asnumpy()
    dist_gsize = all_gsizes/all_gsizes.sum(axis=0,keepdims=1)
    return choice(H.edges, p=dist_gsize)

def whats_happening(event_queue, H, group, params, t):
    u, nu_n, nu_p, alpha, beta, b = params
    p, n = count_prog_nonprog(H, group)

    R_nonprog_grad = nu_n * n
    R_prog_grad = nu_p * p
    R_new_nonprog = mu
    R_conversion_attempt = τ(n, p, alpha, beta, b=b) if n > 0 else 0.
    
    # draw time to next event and event type
    R = R_nonprog_grad + R_conversion_attempt + R_new_nonprog + R_prog_grad
    tau = exponential(scale=1/R)
    
    taulab = ["non-prog graduates", "programmer graduates", "conversion attempt", "new non-programmer"]
    tauR = [R_nonprog_grad/R, R_prog_grad/R, R_conversion_attempt/R, R_new_nonprog/R] 
    which_event = choice(taulab, p=tauR)
    
    if which_event == 'conversion attempt':
        if R_conversion_attempt/R < (1-c(n,p,b=b)):
            next_event = (tau, group, "new programmer", p, n)
        else:
            next_event = (tau, group, "non-prog leaves", p, n)

    next_event = (tau, group, which_event, p, n)

    heapq.heappush(event_queue, next_event)

    return event_queue

#parameters
mu  = 0.5
nu_n = 0.01
nu_p = 0.01
alpha  = 0.01
beta = 0.3 
a = 3.   
b = 8
params = (mu, nu_n, nu_p, alpha, beta, a, b)

#for each simulation
for sims in range(10):
    
    #initial conditions
    event_queue = []
    H = init_hypergraph()
    I0 = 0.01
    I = 0
    Ig = np.zeros((21,21))

    time = 0
    for group, members in enumerate(H.edges.members()): #loop over groups
        # group, members = 0, H.edges.members(0)
        gsize = len(members)
        states_binary = binomial(gsize, I0, size=gsize)
        states = np.where(states_binary == 0, 'non-prog', 'prog')
        for node, state in zip(members, states):
            H.nodes[node]['state'] = state
        event_queue = whats_happening(event_queue, H, group, params, t)
        nb_prog = np.sum(states_binary)
        Ig[nb_prog, gsize-nb_prog] += 1
        I += nb_prog
    
    history = []
    history_pref = []
    history_group = []
    tot_pop = []
    history.append(I/H.num_nodes)
    history_group.append(Ig/np.sum(Ig))
    tot_pop.append(H.num_nodes)
    times = np.zeros(1)
    
    #for each generation
    t_max = 1000
    while time < t_max and len(event_queue) > 0:
        # draw from event queue
        (tau, group, event, p, n) = heapq.heappop(event_queue)
        
        while H.edges.get(group) is None:
            (tau, group, event, p, n) = heapq.heappop(event_queue)

        if event == 'non-programmer graduates' or event == "non-prog leaves":           
            leaving_node = grab(H, group, 'non-prog')
            # Not ideal.
            if leaving_node:
                H.remove_node(leaving_node)
                Ig[p, n]   -= 1
                Ig[p, n-1] += 1
                

        if event == 'programmer graduates':
            leaving_node = grab(H, group, 'prog')
            # Not ideal.
            if leaving_node:
                H.remove_node(leaving_node)
                I -= 1
                Ig[p, n]   -= 1
                Ig[p-1, n] += 1
        
        if event == 'new non-programmer':
            # A bit weird. Although the event is happening on a particular group
            # the new node decide to join another group proportional to gsize.
            # But you know..
            selected_group = pref_attachment_new_node(H)
            new_node = H.num_nodes+1
            H.add_node_to_edge(selected_group, new_node)
            H.nodes[new_node]['state'] = 'non-prog'
            # Now that we have a new programmer in that group, something else might happen.
            # Another students might join the university, another student in the same group
            # might try it too, etc.
            event_queue = whats_happening(event_queue, H, group, params, t)
            Ig[p, n+1] += 1
            Ig[p, n]   -= 1
            history_pref.append(selected_group)

        if event == 'new programmer':
            converting_node = grab(H, group, 'non-prog')
            # Not ideal.
            if converting_node:
                H.nodes[converting_node]['state'] = 'prog'
                I += 1
                Ig[p+1, n] += 1
                Ig[p, n]   -= 1
                # Now that we have a new programmer in that group, something else might happen.
                # Another students might join the university, another student in the same group
                # might try it too, etc.
                event_queue = whats_happening(event_queue, H, group, params, t)

        #update history
        time = time + tau
        tot_pop.append(H.num_nodes)
        history = np.append(history, I / H.num_nodes)
        history_group.append(Ig/np.sum(Ig))
        times = np.append(times, time)
    
    #print time series
    plt.plot(times, history, marker="o", ls='--', color='black', alpha=0.05)
    plt.plot(times, tot_pop, color='midnightblue', alpha=0.5)
    plot_pref(history_pref)
    plot_frac_prog_group(history_group, times)

  

plt.plot(times[-1],history[-1], marker="o", ls='--', color='black', alpha=0.05, label='Simulations')
plt.legend()
plt.ylabel('Frac Programmers')
plt.xlabel('Time')
plt.show()




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

