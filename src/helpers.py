from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_max(history_group, type, t):
    """type: type=1 => non-prog; type=0 => prog"""
    return np.max(np.nonzero(history_group[t])[0 if type == 'prog' else 1])

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


# def plot_sigmoid_mu(n, Ks, mu):   
#     fig, ax = plt.subplots(1,1, figsize=(7,7))
#     for K in Ks:
#         ys = [sigmoid(p, n, K=K, mu=mu) for p in range(20) if sigmoid(p, n, K=K, mu=mu) >= 0]
#         sns.scatterplot(x=range(len(ys)), y=ys, label=f"K={K}", ax=ax)
#     ax.set_xlabel("# programmers")
#     ax.set_ylabel("f(p,n)")
#     plt.title(f"mu={mu}; # non-prog={n}")


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


def wrangle_Ig(Ig_norm, only_prog=True):
        # only_prog=False
        # Ig_norm=history_group
        max_group = 20 if only_prog else 40
        Ig_norm_wrangled = np.zeros((len(Ig_norm), max_group))
        for t in range(len(Ig_norm)):
            # t=13
            num = np.zeros(max_group+1)   # Limit our attention to gsize < 21 for prog
            denum = np.zeros(max_group+1) # because that's what we do in our AME model.
            for gsize in range(1,max_group): 
                p_min = gsize-19 if gsize - 20 >= 0 else 0
                p_max = min([20, gsize])
                for p in range(p_min, p_max):    
                    n = gsize-p
                    # print((gsize, p, n))
            
                    if only_prog: # we only do numerator
                        num[gsize] += (p / gsize) * Ig_norm[t][p, n]
                    
                    denum[gsize] += Ig_norm[t][p, n]
            
            if only_prog:
                weighted_sum = np.where(np.isnan(num[1:]/denum[1:]), 0, num[1:]/denum[1:])
                Ig_norm_wrangled[t,:] = weighted_sum
            else: # fraction of group is just the denum
                Ig_norm_wrangled[t,:] = denum[1:]

        return Ig_norm_wrangled
        

def plot_group_heatmap(Ig_norm_wrangled, only_prog=False, ax=None):
    """return gsize x time heatmap with z=fraction of programmers""" 
    # Ig_norm_wrangled=ig_wrangled
    # only_prog=True
    max_group = len(Ig_norm_wrangled[0,:])-1
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(18,8))
    sns.heatmap(pd.DataFrame(Ig_norm_wrangled).transpose()[:max_group], 
                cmap= "Blues" if only_prog else "Greens", 
                cbar_kws={"label": "frac programmers" if only_prog else "frac gsize"}, ax=ax)
    # labels = [int(item.get_text())+1 for item in ax.get_yticklabels()]
    ax.set_xlabel("Time →")
    ax.set_ylabel("Group size")
    ax.set_xticklabels([]);


def plot_quartet(history, tot_pop, params, history_group, times):
    fig, axes = plt.subplots(2, 2, figsize=(15,10))
    
    ig_wrangled = wrangle_Ig(history_group, only_prog=True)
    ig_wrangled_group = wrangle_Ig(history_group, only_prog=False)
    
    plot_group_heatmap(ig_wrangled, only_prog=True, ax=axes[0,0])
    axes[0,0].set_xlabel("")
    
    plot_group_heatmap(ig_wrangled_group, only_prog=False, ax=axes[0,1])
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
    title += f"\nPI carrying capacity={params[-1]}"
    fig.suptitle(title, fontsize=16)


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

