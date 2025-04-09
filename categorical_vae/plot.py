import numpy as np
import pandas as pd
import tqdm

import matplotlib
matplotlib.rcParams['ps.useafm'] = False
matplotlib.rcParams['pdf.use14corefonts'] = False
matplotlib.rcParams['text.usetex'] = False

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
sns.set_style("white")
sns.set_context("poster")

from pathlib import Path


import torch.nn.functional as F
from pathlib import Path
import tqdm



def plot_decoder_prob_matrices(model, actors_N, actions_A, dates_T, figdir=None):
    """
    Visualize the latent components (one by one) from the decoder's probability matrices in a CVAE model.

    Parameters:
    - model: The CVAE model with a decoder containing `probability_matrices`.
    - actors_N: List of actor names (for sender/receiver axes).
    - actions_A: List of action names (for action axis).
    - dates_T: List of time labels (for time axis).
    - title: Optional string used as a prefix for filenames.
    - figdir: Directory to save plots.
    """
    
    if figdir is None:
        raise ValueError("figdir must be specified to save plots.")
    
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    # Access decoder and its linear projection matrices
    decoder = model.decoder
    prob_matrices = decoder.probability_matrices  # This is a torch.nn.ModuleList

    # Extract and normalize with softmax
    W_s = F.softmax(prob_matrices[0].weight.data.T, dim=1)  # shape: (latent_dim, #senders)
    W_r = F.softmax(prob_matrices[1].weight.data.T, dim=1)  # shape: (latent_dim, #receivers)
    W_a = F.softmax(prob_matrices[2].weight.data.T, dim=1)  # shape: (latent_dim, #actions)
    W_t = F.softmax(prob_matrices[3].weight.data.T, dim=1)  # shape: (latent_dim, #timestamps)


    latent_dim = W_s.shape[0]

    print('Plotting from decoder probability matrices...')
    for k in tqdm.tqdm(range(latent_dim)):
        figname = f'decoder_k_{k}'
        filename = figdir / figname

        plot_component(
            Phi_s_V=W_s[k].cpu().numpy(),       # Sender component
            Phi_r_V=W_r[k].cpu().numpy(),       # Receiver component
            Phi_A=W_a[k].cpu().numpy(),         # Action component
            Theta_T=W_t[k].cpu().numpy(),       # Time component
            actors_N=actors_N,
            actions_A=actions_A,
            dates_T=dates_T,
            cdf_threshold=0.95,
            min_stems=5,
            max_stems=20,
            small_font=8,
            large_font=12,
            title_font=20,
            title=None,
            figsize=(18, 9),
            dpi=400,
            filename=filename,
            legend=False
        )



def plot_component(Phi_s_V, Phi_r_V, Phi_A, Theta_T, 
                   actors_N, actions_A, dates_T,
                   cdf_threshold=0.95, min_stems=5, max_stems=20,
                   small_font=8, large_font=12, title_font=20, title=None,
                   figsize=(18, 9), dpi=400, filename=None, legend=False):
    if filename is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig = plt.figure(figsize=figsize)  # ignore dpi if not serializing plot

    

    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[1, 1, 1],
                           height_ratios=[1, 1])

    # gs.update(hspace=0.05)

    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1], sharey=ax2)
    ax4 = plt.subplot(gs[1,2], sharey=ax2)

    gs.update(wspace=0.075, hspace=0.2)

    T_N = Theta_T.copy()
    T_N /= T_N.sum()

    S_N = Phi_s_V.copy()
    S_N /= S_N.sum()

    R_N = Phi_r_V.copy()
    R_N /= R_N.sum()

    A_N = Phi_A.copy()
    A_N /= A_N.sum()


    #### TIME STEPS ####
    elem1, = ax1.plot(T_N, 'o-', color=sns.color_palette("Blues")[-1])
    ax1.fill_between(range(T_N.size), T_N,  color=sns.color_palette("Blues")[-4], alpha=0.5)
    # ax1.xaxis.tick_top()

    # ax1.set_xticks(np.arange(1, 15) * 6 - 2)
    xticks = np.linspace(3, T_N.size - 3, num=10, dtype=int)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([dates_T[int(x)] for x in xticks], rotation=0, fontsize=large_font)

    ax1.set_xlim(-1, max(xticks) + 10)
    ax1.set_ylim(0, max(ax1.get_yticks()))
    plt.setp(ax1.get_yticklabels(), fontsize=small_font)

    if title is not None:
        ax1.set_title(title, fontsize=title_font)

    

    #### SENDERS ####
    ranked = S_N.argsort()
    ranked = ranked[::-1][:-1]  # remove nan
    ranked_vals = [S_N[i] for i in ranked]
    ranked_names = [actors_N[i] for i in ranked]
    max_n = np.searchsorted(np.cumsum(ranked_vals), cdf_threshold)
    top_n = min(max(max_n, max_stems), min_stems)
    
    markerline2, stemlines2, baseline2 = ax2.stem(range(top_n), ranked_vals[:top_n])
    plt.setp(baseline2, 'color', sns.color_palette("Reds")[-1])
    plt.setp(stemlines2, 'color', sns.color_palette("Reds")[-1])
    plt.setp(markerline2, 'color', sns.color_palette("Reds")[-1])
    ax2.set_xticks(range(top_n))
    ax2.set_xlim(-0.25, top_n - 1  + .25)
    ax2.set_xticklabels(ranked_names[:top_n], rotation=90, fontsize=large_font, ha='right')
    plt.setp(ax2.get_yticklabels(), fontsize=small_font)

    #### RECEIVERS ####
    ranked = R_N.argsort()
    ranked = ranked[::-1][:-1]  # remove nan
    ranked_vals = [R_N[i] for i in ranked]
    ranked_names = [actors_N[i] for i in ranked]
    top_n = np.searchsorted(np.cumsum(ranked_vals), cdf_threshold)
    top_n = np.clip(top_n, min_stems, max_stems)

    markerline3, stemlines3, baseline3 = ax3.stem(range(top_n), ranked_vals[:top_n])
    plt.setp(baseline3, 'color', sns.color_palette("Greens")[-1])
    plt.setp(stemlines3, 'color', sns.color_palette("Greens")[-1])
    plt.setp(markerline3, 'color', sns.color_palette("Greens")[-1])
    ax3.set_xticks(range(top_n))
    ax3.set_xlim(-0.25, top_n - 1  + .25)
    ax3.set_xticklabels(ranked_names[:top_n], rotation=90, fontsize=large_font, ha='right')
    plt.setp(ax3.get_yticklabels(), visible=False)

        #### ACTIONS ####
    ranked = A_N.argsort()
    ranked = ranked[::-1][:-1]  # remove nan
    ranked_vals = [A_N[i] for i in ranked]
    ranked_names = [actions_A[i] for i in ranked]
    top_n = np.searchsorted(np.cumsum(ranked_vals), cdf_threshold)
    top_n = np.clip(top_n, min_stems, max_stems)

    markerline4, stemlines4, baseline4 = ax4.stem(range(top_n), ranked_vals[:top_n])
    plt.setp(baseline4, 'color', sns.color_palette("Purples")[-1])
    plt.setp(stemlines4, 'color', sns.color_palette("Purples")[-1])
    plt.setp(markerline4, 'color', sns.color_palette("Purples")[-1])
    ax4.set_xticks(range(top_n))
    ax4.set_xlim(-0.25, top_n - 1  + .25)
    ax4.set_xticklabels(ranked_names[:top_n], rotation=90, fontsize=large_font, ha='right')
    plt.setp(ax4.get_yticklabels(), visible=False)

    # plt.subplots_adjust(hspace=0.35)
    elems = [elem1, markerline2, markerline3, markerline4]
    labels = ['time steps', 'senders', 'receivers', 'actions']
    if legend:
        plt.figlegend(elems, labels, loc=(0.83, 0.8), fontsize=large_font)  #(up is right, up is up)
    # plt.figlegend(elems, labels, loc=1, fontsize=large_font)

    if filename is not None:
        if filename.suffix != '.pdf':
            filename = filename.with_suffix('.pdf')
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()