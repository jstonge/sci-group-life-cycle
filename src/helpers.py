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

def plot_group_dist(H):
    count_members = Counter([len(e) for e in H.edges.members()])
    df = pd.DataFrame({"n": count_members.values(), "k": count_members.keys()})
    sns.barplot(x="k", y="n", data=df, color="darkblue", alpha=0.5)
    plt.xlabel("group size")
    plt.ylabel("count")

def plot_pref(history_pref):
    count_df = pd.DataFrame(Counter(history_pref).most_common(50), columns=['group', '# selected'])\
                 .assign(group = lambda x: x.group.astype(str))
    
    fig, ax = plt.subplots(1,1,figsize=(5, 10))
    sns.barplot(y="group", x="# selected", data=count_df, ax=ax)

def plot_frac_prog_group(history_group, times, ax=None, out=None):
    history_group_wrangled = np.zeros((len(times), (40//2)+1))
    for t in range(len(times)):
        out = np.zeros((40//2)+1)
        for gsize in range((40//2)+1):
            for p in range(gsize):
                n = gsize-p
                out[gsize] += history_group[t][p, n]
    
        history_group_wrangled[t,:] = out

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,5))

    sns.lineplot(x=times, y=[history_group_wrangled[t][1] for t in range(len(times))], 
            color='red', alpha=0.95, ax=ax, label="gsize=1")
    sns.lineplot(x=times, y=[history_group_wrangled[t][3] for t in range(len(times))], 
            color='black', alpha=0.55, ax=ax, label="gsize=3")
    sns.lineplot(x=times, y=[history_group_wrangled[t][5] for t in range(len(times))], 
            color='blue', alpha=0.55, ax=ax, label="gsize=5")
    sns.lineplot(x=times, y=[history_group_wrangled[t][7] for t in range(len(times))], 
            color='orange', alpha=0.55, ax=ax, label="gsize=7")
    sns.lineplot(x=times, y=[history_group_wrangled[t][10] for t in range(len(times))], 
            color='green', alpha=0.55, ax=ax, label="gsize=10")
    sns.lineplot(x=times, y=[history_group_wrangled[t][12] for t in range(len(times))], 
            color='cyan', alpha=0.55, ax=ax, label="gsize=12")
    sns.lineplot(x=times, y=[history_group_wrangled[t][20] for t in range(len(times))], 
            color='purple', alpha=0.55, ax=ax, label="gsize=20")
    ax.set_ylabel("frac programmers")
    ax.set_xlabel("Time")
    plt.legend()

    if out is not None:
        plt.savefig("test_sims.png")

def plot_group_size(history_group, times, ax=None, out=None):
    history_group_wrangled = np.zeros((len(times), 40+1))
    for t in range(len(times)):
        out = np.zeros(40+1)
        for gsize in range(40+1):
            for p in range(gsize):
                n = gsize-p
                out[gsize] += history_group[t][p, n]
        history_group_wrangled[t,:] = out

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,5))

    sns.lineplot(x=times, y=[history_group_wrangled[t][1] for t in range(len(times))], 
            color='red', alpha=0.95, ax=ax, label="gsize=1")
    sns.lineplot(x=times, y=[history_group_wrangled[t][3] for t in range(len(times))], 
            color='black', alpha=1, ax=ax, label="gsize=3")
    sns.lineplot(x=times, y=[history_group_wrangled[t][5] for t in range(len(times))], 
            color='blue', alpha=0.55, ax=ax, label="gsize=5")
    sns.lineplot(x=times, y=[history_group_wrangled[t][7] for t in range(len(times))], 
            color='orange', alpha=0.55, ax=ax, label="gsize=7")
    sns.lineplot(x=times, y=[history_group_wrangled[t][10] for t in range(len(times))], 
            color='green', alpha=0.55, ax=ax, label="gsize=10")
    sns.lineplot(x=times, y=[history_group_wrangled[t][12] for t in range(len(times))], 
            color='cyan', alpha=0.55, ax=ax, label="gsize=12")
    sns.lineplot(x=times, y=[history_group_wrangled[t][20] for t in range(len(times))], 
            color='purple', alpha=0.55, ax=ax, label="gsize=20")
    sns.lineplot(x=times, y=[history_group_wrangled[t][30] for t in range(len(times))], 
            color='lightgrey', ax=ax, alpha=0.85, label="gsize=30")
    sns.lineplot(x=times, y=[history_group_wrangled[t][39] for t in range(len(times))], 
            color='pink', alpha=0.55, ax=ax, label="gsize=39")
    ax.set_ylabel("# groups")
    ax.set_xlabel("Time")
    plt.legend()

    if out is not None:
        plt.savefig("test_sims.png")

