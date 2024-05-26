## Custom Visualizations

import networkx as nx
import numpy as np


###############################################################################
# visualization utils
###############################################################################

def shrink_v1(rx, ry, factor=11):
    """This function is used to make the walks smooth."""

    rx = np.array(rx)
    ry = np.array(ry)

    rx = 0.75 * rx + 0.25 * rx.mean()
    ry = 0.75 * ry + 0.25 * ry.mean()

    last_node = rx.shape[0] - 1
    concat_list_x = [np.linspace(rx[0], rx[0], 5)]
    concat_list_y = [np.linspace(ry[0], ry[0], 5)]
    for j in range(last_node):
        concat_list_x.append(np.linspace(rx[j], rx[j + 1], 5))
        concat_list_y.append(np.linspace(ry[j], ry[j + 1], 5))
    concat_list_x.append(np.linspace(rx[last_node], rx[last_node], 5))
    concat_list_y.append(np.linspace(ry[last_node], ry[last_node], 5))

    rx = np.concatenate(concat_list_x)
    ry = np.concatenate(concat_list_y)

    filt = np.exp(-np.linspace(-2, 2, factor) ** 2)
    filt = filt / filt.sum()

    rx = np.convolve(rx, filt, mode='valid')
    ry = np.convolve(ry, filt, mode='valid')

    return rx, ry


###############################################################################
# 2D visualization
###############################################################################
def relevance_vis_2d_v1(
        ax,
        relevances,
        atomic_numbers,
        pos,
        graph,
        shrinking_factor=11,
):

    ####################################################################################################################
    # Utils
    ####################################################################################################################

    def _iterate_over_all_walks(ax, relevances):

        # visualization settings
        selfloopwidth = 0.32
        linewidth = 4.
        max_rel=-1
        new_relevances = []

        for walk, rel in relevances:
            max_rel = max(abs(rel),max_rel)
        
        for idx , (walk, rel) in enumerate(relevances):
            new_relevances.append((walk, rel/max_rel))

        #print(new_relevances)

        # start iteration over walks
        for walk_id, (walk, relevance) in enumerate(new_relevances):
            # get walk color
            color = '#66ff66' if relevance < 0 else '#ff66ff'
            # get opacity
            alpha = abs(relevance)
            #alpha = 0.9
            # split position vector in x and y part
            rx = np.array([pos[node][0] for node in walk])
            ry = np.array([pos[node][1] for node in walk])
            # plot self loops
            for i in range(len(rx) - 1):
                if rx[i] == rx[i + 1] and ry[i] == ry[i + 1]:
                    rx_tmp = rx[i] + selfloopwidth * np.cos(np.linspace(0, 2 * np.pi, 16))
                    ry_tmp = ry[i] + selfloopwidth * np.sin(np.linspace(0, 2 * np.pi, 16))
                    ax.plot(rx_tmp, ry_tmp, color=color, alpha=alpha, lw=linewidth, zorder=1.)
            # plot walks
            rx, ry = shrink_v1(rx, ry, shrinking_factor)
            ax.plot(rx, ry, color=color, alpha=alpha, lw=linewidth, zorder=1.)
        return ax

    ####################################################################################################################
    # Main function code
    ####################################################################################################################

    # plot walks
    ax = _iterate_over_all_walks(ax, relevances)
    # prepare molecular graph
    atom_names_dict = {35: "Br",16: "S",51: "Sb",4: "Be",14: "Si",64: "Gd",42: "Mo",22: "Ti",19: "K",49: "In",12: "Mg",29: "Cu",33: "As",80: "Hg",60: "Nd",43: "Tc",48: "Cd",15: "P",6: "C",46: "Pd",23: "V",28: "Ni",47: "Ag",38: "Sr",32: "Ge",24: "Cr",1: "H",25: "Mn",79: "Au",20: "Ca",66: "Dy",56: "Ba",9: "F",78: "Pt",40: "Zr",17: "Cl",11: "Na",30: "Zn",13: "Al",81: "Tl",3: "Li",82: "Pb",8: "O",5: "B",7: "N",27: "Co",0: "*",70: "Yb",53: "I",83: "Bi",34: "Se",26: "Fe",50: "Sn"}
    names = [atom_names_dict[Z.item()] for i, Z in enumerate(atomic_numbers)]
    G = nx.from_numpy_array(graph)
    # plot atoms
    collection = nx.draw_networkx_nodes(G, pos, node_color="w", node_size=1000)
    collection.set_zorder(2.)
    # plot bonds
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color="w",
        width=1,
        #style="dotted",
        node_size=300
    )
    # plot atom types
    pos_labels = pos - np.array([0.02, 0.05])
    nx.draw_networkx_labels(G, pos_labels, {i: name for i, name in enumerate(names)}, font_size=40)
    return ax
