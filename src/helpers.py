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


def plot_group_size(history_group, times, ax=None, gsizes=None, out=None):
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

    if gsizes is None:
        gsizes = [3,4,6, 8,10,13,17,22,25,33,39]
    assert len(gsizes) <= len(sns.palettes.SEABORN_PALETTES['dark']), "too many gsize"
    for gsize, col in zip(gsizes, sns.palettes.SEABORN_PALETTES['dark'][:len(gsizes)]):
        sns.lineplot(x=times, y=[history_group_wrangled[t][gsize] for t in range(len(times))], 
        color=col, alpha=0.95, ax=ax, label=f"gsize={gsize}")
    ax.set_ylabel("# groups")
    ax.set_xlabel("Time")
    plt.legend()

    if out is not None:
        plt.savefig("test_sims.png")

