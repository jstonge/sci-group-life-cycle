from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def count_prog_nonprog(H, group):
    """return (p, n)"""
    states_count = Counter(H.nodes[n]['state'] for n in H.edges.members(group))
    if len(states_count) == 2:
        return states_count.values()
    elif states_count.get('prog') is None:
        return 0, list(states_count.values())[0]
    else:
        return list(states_count.values())[0], 0

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

