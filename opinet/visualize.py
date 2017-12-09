"""
Utility module to graph static graphs and dynamic graphs by animation.
"""

import os
import commands
import shutil

import numpy as np
import igraph as ig

from util import mat_to_edge_list

def plot_graph(E_mat, stances=None, actions=None, scale_nodes=None,  
               layout_seed=None, layout_exact=False, margin=None,
               file_name=None):
    """
    Plots a single static graph.
    """
    n = E_mat.shape[0]
    edge_list = mat_to_edge_list(E_mat)
    G = ig.Graph(n=n, edges=edge_list, directed=True)

    visual_style = {}
    visual_style["edge_arrow_size"] = 0.5
    visual_style["edge_arrow_width"] = 0.5
    visual_style["edge_width"] = 0.5
    visual_style["vertex_size"] = 8
    visual_style["dpi"] = 300
    visual_style["margin"] = 20 if margin is None else margin

    if layout_seed is None:
        visual_style["layout"] = G.layout("kk")
    else:
        visual_style["layout"] = (layout_seed if layout_exact else 
                                  G.layout_kamada_kawai(maxiter=500, seed=
                                  layout_seed))
    
    if not scale_nodes is None:
        axis = 0 if scale_nodes == "indegree" else 1
        degrees = list(np.nansum(E_mat, axis=axis))
        visual_style["vertex_size"] = degrees

    if not stances is None:
        stances_int = ((np.around(stances, 2) + 1) * 100).astype(int)
        pal = ig.GradientPalette("red", "blue", 201)
        vertex_colors = [pal.get(stances_int[i]) for i in range(n)]
        visual_style["vertex_color"] = vertex_colors
        visual_style["vertex_frame_color"] = vertex_colors
    else:
        visual_style["vertex_color"] = "black"
        visual_style["vertex_frame_color"] = "black"
        visual_style["height"] = 4000
        visual_style["width"] = 4000

    if not actions is None:
        actions_int = ((np.around(actions, 2) + 1) * 100).astype(int)
        edges_ordered = [e.tuple for e in G.es]
        ordered_actions = [actions_int[j] for (i, j) in edges_ordered]
        edge_colors = [pal.get(ordered_actions[i]) for i in range(len(
                       ordered_actions))]
        G.es["color"] = edge_colors

    if not file_name is None:
        export_path = os.path.join("..", "plots", file_name + ".png")
        G_plot = ig.plot(G, target=export_path, res=6000, **visual_style)
        return G_plot, G, visual_style

    return ig.plot(G, **visual_style), G, visual_style

def plot_dynamic_graph(G, video_name, stances=None, actions=None, 
                       scale_nodes=None, margin=None):
    """
    Produces an animation of a time-dynamic graph through repeated calls to 
    plot_graph().
    """
    T = G.shape[0]
    assert(T < 1000)

    # names of snapshots
    video_dir_full = os.path.join("..", "plots", video_name)
    shutil.rmtree(video_dir_full)
    os.makedirs(video_dir_full)
    file_names_root = os.path.join(video_name, video_name)
    file_names = [file_names_root + "_" + (3 - len(str(i))) * "0" + str(i) for 
                  i in range(T)]

    # initial configuration
    stances_in = None if stances is None else stances[0]
    actions_in = None if actions is None else actions[0]
    G_plot, G_i, visual_style = plot_graph(G[0], stances=stances_in, 
        actions=actions_in, scale_nodes=scale_nodes, margin=margin, 
        file_name=file_names[0])

    # update in each time step
    for t in range(1, T):
        stances_in = None if stances is None else stances[t]
        actions_in = None if actions is None else actions[t]
        G_plot, G_i, visual_style = plot_graph(G[t], stances=stances_in, 
            actions=actions_in, scale_nodes=scale_nodes, 
            layout_seed=visual_style["layout"], layout_exact=False, 
            margin=margin, file_name=file_names[t])

    # create video
    in_path = os.path.join("..", "plots", video_name, video_name + "_" +
                           "%03d.png")
    out_path = os.path.join("..", "plots", video_name, video_name + ".mp4")
    call_str = ("ffmpeg -r 6 -i " + in_path + 
                " -y -pix_fmt yuv420p -r 6 -crf 18 -q:v 0 " + out_path)
    commands.getoutput(call_str)
