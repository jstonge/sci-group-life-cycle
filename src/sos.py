import heapq
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgi
from numpy.random import binomial, choice, exponential, rand, logistic

from helpers import count_prog_nonprog, grab, plot_frac_prog_group, plot_group_size


def count_prog_nonprog(H, group):
    """return (p, n)"""
    if H.edges.get(group) is None:
       return 0, 0
    else:
        states_count = Counter(H.nodes[n]['state'] for n in H.edges.members(group))    
        if len(states_count) == 2:
            return states_count.get('prog'), states_count.get('non-prog')
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

def adding_new_non_prog(H, group, tot_nodes):
    new_node =  tot_nodes + 1
    if H.edges.get(group) is None:
        # print(f"group {group} came back to life.")
        H.add_edge([new_node], id=group)
    else:
        H.add_node_to_edge(group, new_node)

    H.nodes[new_node]['state'] = 'non-prog'

def whats_happening(event_queue, H, group, params, t):
    mu, nu_n, nu_p, alpha, beta, b, pa = params
    p, n = count_prog_nonprog(H, group)

    R_nonprog_grad = nu_n * n
    R_prog_grad = nu_p * p
    R_new_nonprog = mu * (1 + (n+p) * pa)
    R_conversion_attempt = τ(n, p, alpha, beta, b=b) if n > 0 else 0.
    
    # draw time to next event and event type
    R = R_nonprog_grad + R_conversion_attempt + R_new_nonprog + R_prog_grad
    tau = exponential(scale=1/R)
    
    taulab = ["non-prog graduates", "programmer graduates", "conversion attempt", "new non-programmer"]
    tauR = [R_nonprog_grad/R, R_prog_grad/R, R_conversion_attempt/R, R_new_nonprog/R] 
    which_event = choice(taulab, p=tauR)
    
    # Ok, now the new tau is added to time
    next_event = (t+tau, group, which_event, p, n)
    
    if which_event == 'conversion attempt':
        if R_conversion_attempt/R < (1-c(n,p,b=b)):
            next_event = (t+tau, group, "new programmer", p, n)
        else:
            next_event = (t+tau, group, "non-prog leaves", p, n)

    heapq.heappush(event_queue, next_event)

    return event_queue

def main():
    params = (0.1, 0.01, 0.01, 0.01, 0.1, .5, 1)
    I0 = 0.1

    #initial conditions
    event_queue = []
    max_group_size = 20
    H = init_hypergraph(max_group_size=max_group_size)
    I = 0
    Ig = np.zeros((max_group_size*10,max_group_size*10))
    t = 0
    # total nodes dead or alive to have unique node identifier.
    tot_nodes = H.num_nodes
    # we start out we a hypergraph where hyperedges == groups
    # they are non-overlapping at the moment. all_gsize > 0.
    for group, members in enumerate(H.edges.members()): #loop over groups
        gsize = len(members)
        states_binary = binomial(1, I0, size=gsize)
        p = np.sum(states_binary)
        n = gsize - p
        states = np.where(states_binary == 0, 'non-prog', 'prog')
        for node, state in zip(members, states):
            H.nodes[node]['state'] = state
        event_queue = whats_happening(event_queue, H, group, params, t)
        Ig[p, n] += 1
        I += p

    assert all([Ig[q[-2], q[-1]] != 0 for q in event_queue]), "Ig should never be zero when there are non-prog/prog"

    # what's in the event queue?
    # Counter(e[2] for e in event_queue)
    # is the time alright?
    # [e[0] for e in event_queue]

    history = []
    history_group = []
    tot_pop = []
    history.append(I/H.num_nodes)
    # history_group_prog.append(Ig/np.sum(Ig))
    history_group.append(Ig / np.sum(Ig))
    tot_pop.append(H.num_nodes)
    times = np.zeros(1)

    #for each generation
    t_max = 20
    
    #for each generation
    while t < t_max:

        # draw from event queue
        # here is use time_tau because in whats_happening I always
        # add time + tau. So this is not just tau, this is current time.
        (time, group, event, p, n) = heapq.heappop(event_queue)  

        t=time

        # while H.edges.get(group) is None:
        #     (time, group, event, p, n) = heapq.heappop(event_queue)
        # Ig[p, n]  -= 1

        if event == 'non-programmer graduates' or event == "non-prog leaves":           
            leaving_node = grab(H, group, 'non-prog')
            # Not ideal.
            assert leaving_node is not None, "node is None"
            if leaving_node:
                H.remove_node(leaving_node)
                assert n > 0
                Ig[p, n]   -= 1
                Ig[p, n-1] += 1
                
        if event == 'programmer graduates':
            leaving_node = grab(H, group, 'prog')
            assert leaving_node is not None, "node is None"
            # Not ideal.
            if leaving_node:
                H.remove_node(leaving_node)
                I -= 1
                assert p > 0
                Ig[p, n]   -= 1
                Ig[p-1, n] += 1
    
        if event == 'new non-programmer':
            adding_new_non_prog(H, group, tot_nodes)
            tot_nodes += 1
            # assert n < max_group_size, f"We currently have {p} prog and {n} non progs in group {group}"
            Ig[p, n]  -= 1
            Ig[p, n+1] += 1
        
        if event == 'new programmer':
            converting_node = grab(H, group, 'non-prog')
            assert converting_node is not None, "node is None"
            # Not ideal.
            if converting_node:
                H.nodes[converting_node]['state'] = 'prog'
                I += 1
                # assert p < max_group_size and n > 0
                Ig[p, n]     -= 1
                Ig[p+1, n-1] += 1

        event_queue = whats_happening(event_queue, H, group, params, t)

        # update history
        tot_pop.append(H.num_nodes)
        history = np.append(history, I / H.num_nodes)
        history_group.append(Ig / np.sum(Ig))
        # history_group_prog.append(Ig/np.sum(Ig))
        times = np.append(times, time)


    # sns.set_style('whitegrid')
    def plot_quartet():
        fig, axes = plt.subplots(2, 2, figsize=(15,10))
        plot_frac_prog_group(history_group, times, ax=axes[0,0])
        axes[0,0].set_xlabel("")
        gsizes = [4,5,6,7,8,9,11,12,13]
        plot_group_size(history_group, times, ax=axes[0,1], gsizes=gsizes)
        axes[0,1].set_xlabel("")

        sns.lineplot(x=times, y=history, color='black', ax=axes[1,0])
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Frac Programmers')
        axes[1,0].set_ylim(0, np.max(history)+0.2)

        sns.lineplot(x=times, y=tot_pop, color='midnightblue', alpha=1., ax=axes[1,1])
        axes[1,1].set_ylabel('Total Population')
        axes[1,1].set_xlabel('Time')

        param_lab = ['mu', 'nu_n', 'nu_p', 'alpha', 'beta', 'b']
        title=';'.join([f'{lab}={val}' for lab, val in zip(param_lab, params)])
        title += f"\nPreferential attachment={params[-1]}"
        fig.suptitle(title, fontsize=16)

    plot_quartet()
    
    # plt.savefig(f"../figs/summary_pa{params[-1]}.png")

    # individual plots

    def wrangle_Ig(Ig_norm, only_prog=True):
        max_group = 20 if only_prog else 40
        Ig_norm_wrangled = np.zeros((len(times), max_group))
        for t in range(len(Ig_norm)):
            num = np.zeros(max_group+1)   # Limit our attention to gsize < 21 for prog
            denum = np.zeros(max_group+1) # because that's what we do in our AME model.
            for gsize in range(max_group+1): 
                for p in range(gsize):
                    n = gsize-p
                    if only_prog:
                        num[gsize] += (p / gsize) * Ig_norm[t][p, n]
                    denum[gsize] += Ig_norm[t][p, n]
            
            if only_prog:
                weighted_sum = np.where(np.isnan(num[1:]/denum[1:]), 0, num[1:]/denum[1:])
                Ig_norm_wrangled[t,:] = weighted_sum
            else:
                Ig_norm_wrangled[t,:] = denum[1:]

        return Ig_norm_wrangled

        
    def plot_group_heatmap(Ig_norm, times, only_prog=False, ax=None, out=None):
        """return gsize x time heatmap with z=fraction of programmers""" 
        # Ignore gsize=0 b/c they'll be always zero
        # Ig_norm=history_group
        # only_prog=True
        max_group = 20 if only_prog else 40
        Ig_norm_wrangled = wrangle_Ig(Ig_norm, only_prog=only_prog)

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(18,8))
        sns.heatmap(pd.DataFrame(Ig_norm_wrangled).transpose(), cmap= "Blues" if only_prog else "Greens", 
                    cbar_kws={"label": "frac programmers"}, ax=ax)
        ax.set_yticklabels([_ for _ in range(1,max_group+1)]);
        ax.set_xlabel("Time →")
        ax.set_ylabel("Group size")
        ax.set_xticklabels("");
        # plt.savefig("update_frac_group.png")
    
    # fig, ax = plt.subplots(1,1,figsize=(15,12))
    # gsizes = [6, 8, 11, 13, 15]
    # plot_group_size(history_group, times, ax=ax, gsizes=gsizes)

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(30,20))
    plot_group_heatmap(history_group, times, only_prog=True, ax=ax1)
    plot_group_heatmap(history_group, times, ax=ax2)
    plt.savefig("update_frac_group.png")



main()




