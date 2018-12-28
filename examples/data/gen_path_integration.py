#!/usr/bin/env python

"""
Create generic LPU and simple pulse input signal.
"""

from itertools import product
import sys

import numpy as np
import h5py
import networkx as nx
import matplotlib
matplotlib.use('agg')
import pylab as plt

from funcs import *
from path_integration import trials

def create_lpu_graph():

    '''
    Naming convention
    Neurons: neuron type _ neuron #
    Synapse: from neuron _ to neuron
    WeightedSum: ws _ to neuron
    '''

    # Neuron ids are between 0 and the total number of neurons:
    G = nx.MultiDiGraph()

    # heading (input) to tl2
    for i in range(tl2_prefs.shape[0]):
        id = "tl2_" + str(i)
        name = "tl2_" + str(i)
        G.add_node(id,
                **{'class': 'TL2',
                    'name': name + '_s',
                    'init_r': 0., # value
                    'prefs': float(tl2_prefs[i]),
                    'slope': tl2_slope,
                    'bias': tl2_bias
                    })

        # Synapse + WeightedSum to cl1
        G.add_node(id + '_cl1_' + str(i),
                **{'class': 'Synapse',
                    'name': id + '_cl1_' + str(i) + '_s',
                    'init_individual_input': 0.,
                    'w': -1.
                    })

        G.add_node('ws_cl1_' + str(i),
                **{'class': 'WeightedSum',
                    'name': 'ws_cl1_' + str(i) + '_s',
                    'init_I': 0.,
                    'dummy': 0.
                    })

        G.add_node('cl1_' + str(i),
                **{'class': 'CL1',
                    'name': 'cl1_' + str(i) + '_s',
                    'init_r': 0.,
                    'slope': 3.0,
                    'bias': -0.5
                    })

        # Add edges

        G.add_edge(id, id + '_cl1_' + str(i))
        G.add_edge(id + '_cl1_' + str(i), 'ws_cl1_' + str(i))
        G.add_edge('ws_cl1_' + str(i), 'cl1_' + str(i))

    # from cl1 to tb1, and tb1 to tb1
    for i in range(W_CL1_TB1.shape[0]):
        G.add_node('tb1_' + str(i),
                **{'class': 'TB1',
                    'name': 'tb1_' + str(i) + '_s',
                    'init_r': 0.,
                    'slope': 5.0,
                    'bias': 0.
                    })

        G.add_node('ws_tb1_' + str(i),
                **{'class': 'WeightedSum',
                    'name': 'ws_tb1_' + str(i) + '_s',
                    'init_I': 0.,
                    'dummy': 0.
                    })

        G.add_edge('ws_tb1_' + str(i), 'tb1_' + str(i))

        for j in range(W_TB1_TB1.shape[0]):
            G.add_node('tb1_' + str(i) + '_tb1_' + str(j),
                    **{'class': 'Synapse',
                        'name': 'tb1_' + str(i) + '_tb1_' + str(j) + '_s',
                        'init_individual_input': 0.,
                        'w': -float(W_TB1_TB1[j][i] * prop_tb1)
                        })

            G.add_edge('tb1_' + str(i), 'tb1_' + str(i) + '_tb1_' + str(j))
            G.add_edge('tb1_' + str(i) + '_tb1_' + str(j), 'ws_tb1_' + str(j))

        for j in range(W_CL1_TB1.shape[1]):
            w = W_CL1_TB1[i][j] * prop_cl1

            G.add_node('cl1_' + str(j) + '_tb1_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'cl1_' + str(j) + '_tb1_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('cl1_' + str(j), 'cl1_' + str(j) + '_tb1_' + str(i))
            G.add_edge('cl1_' + str(j) + '_tb1_' + str(i), 'ws_tb1_' + str(i))

    # tn2 [L, R]
    G.add_node('tn2_0',
            **{'class': 'TN2',
                'name': 'tn2_0_s',
                'init_r': 0.,
                'prefs': float(tn_prefs)
                })

    G.add_node('tn2_1',
            **{'class': 'TN2',
                'name': 'tn2_1_s',
                'init_r': 0.,
                'prefs': -float(tn_prefs)
                })

    #cpu4
    for i in range(W_TN_CPU4.shape[0]):
        G.add_node('cpu4_' + str(i),
                **{'class': 'CPU4',
                    'name': 'cpu4_' + str(i) + '_s',
                    'init_r': 0.,
                    'init_V': 0.5,
                    'slope': 5.0,
                    'bias': 2.5,
                    'cpu4_mem_gain': cpu4_mem_gain
                    })

        G.add_node('ws_cpu4_' + str(i),
                **{'class': 'WeightedSum',
                    'name': 'ws_cpu4_' + str(i) + '_s',
                    'init_I': 0.,
                    'dummy': 0.
                    })

        G.add_edge('ws_cpu4_' + str(i), 'cpu4_' + str(i))

        for j in range(W_TN_CPU4.shape[1]):
            w = W_TN_CPU4[i][j]
            G.add_node('tn2_' + str(j) + '_cpu4_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'tn2_' + str(j) + '_cpu4_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('tn2_' + str(j), 'tn2_' + str(j) + '_cpu4_' + str(i))
            G.add_edge('tn2_' + str(j) + '_cpu4_' + str(i), 'ws_cpu4_' + str(i))

        for j in range(W_TB1_CPU4.shape[1]):
            w = -W_TB1_CPU4[i][j]
            G.add_node('tb1_' + str(j) + '_cpu4_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'tb1_' + str(j) + '_cpu4_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('tb1_' + str(j), 'tb1_' + str(j) + '_cpu4_' + str(i))
            G.add_edge('tb1_' + str(j) + '_cpu4_' + str(i), 'ws_cpu4_' + str(i))


    #Pontin
    for i in range(W_CPU4_pontin.shape[0]):
        G.add_node('pontin_' + str(i),
                **{'class': 'Pontin',
                    'name': 'pontin_' + str(i) + '_s',
                    'init_r': 0.,
                    'slope': 5.,
                    'bias': 2.5
                    })

        G.add_node('ws_pontin_' + str(i),
                **{'class': 'WeightedSum',
                    'name': 'ws_pontin_' + str(i) + '_s',
                    'init_I': 0.,
                    'dummy': 0.
                    })

        G.add_edge('ws_pontin_' + str(i), 'pontin_' + str(i))

        for j in range(W_CPU4_pontin.shape[1]):
            w = W_CPU4_pontin[i][j]
            G.add_node('cpu4_' + str(j) + '_pontin_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'cpu4_' + str(j) + '_pontin_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('cpu4_' + str(j), 'cpu4_' + str(j) + '_pontin_' + str(i))
            G.add_edge('cpu4_' + str(j) + '_pontin_' + str(i), 'ws_pontin_' + str(i))

    #CPU1a
    for i in range(W_TB1_CPU1a.shape[0]):
        G.add_node('cpu1a_' + str(i),
                **{'class': 'CPU1a',
                    'name': 'cpu1a_' + str(i) + '_s',
                    'init_r': 0.,
                    'slope': 7.5,
                    'bias': -1.
                    })

        G.add_node('ws_cpu1a_' + str(i),
                **{'class': 'WeightedSum',
                    'name': 'ws_cpu1a_' + str(i) + '_s',
                    'init_I': 0.,
                    'dummy': 0.
                    })

        G.add_edge('ws_cpu1a_' + str(i), 'cpu1a_' + str(i))

        for j in range(W_TB1_CPU1a.shape[1]):
            w = -1. * W_TB1_CPU1a[i][j]
            G.add_node('tb1_' + str(j) + '_cpu1a_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'tb1_' + str(j) + '_cpu1a_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('tb1_' + str(j), 'tb1_' + str(j) + '_cpu1a_' + str(i))
            G.add_edge('tb1_' + str(j) + '_cpu1a_' + str(i), 'ws_cpu1a_' + str(i))

        for j in range(W_CPU4_CPU1a.shape[1]):
            w = 0.5 * W_CPU4_CPU1a[i][j]
            G.add_node('cpu4_' + str(j) + '_cpu1a_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'cpu4_' + str(j) + '_cpu1a_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('cpu4_' + str(j), 'cpu4_' + str(j) + '_cpu1a_' + str(i))
            G.add_edge('cpu4_' + str(j) + '_cpu1a_' + str(i), 'ws_cpu1a_' + str(i))

        for j in range(W_pontin_CPU1a.shape[1]):
            w = -0.5 * W_pontin_CPU1a[i][j]
            G.add_node('pontin_' + str(j) + '_cpu1a_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'pontin_' + str(j) + '_cpu1a_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('pontin_' + str(j), 'pontin_' + str(j) + '_cpu1a_' + str(i))
            G.add_edge('pontin_' + str(j) + '_cpu1a_' + str(i), 'ws_cpu1a_' + str(i))

    #CPU1b
    for i in range(W_TB1_CPU1b.shape[0]):
        G.add_node('cpu1b_' + str(i),
                **{'class': 'CPU1b',
                    'name': 'cpu1b_' + str(i) + '_s',
                    'init_r': 0.,
                    'slope': 7.5,
                    'bias': -1.
                    })

        G.add_node('ws_cpu1b_' + str(i),
                **{'class': 'WeightedSum',
                    'name': 'ws_cpu1b_' + str(i) + '_s',
                    'init_I': 0.,
                    'dummy': 0.
                    })

        G.add_edge('ws_cpu1b_' + str(i), 'cpu1b_' + str(i))

        for j in range(W_TB1_CPU1b.shape[1]):
            w = -1. * W_TB1_CPU1b[i][j]
            G.add_node('tb1_' + str(j) + '_cpu1b_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'tb1_' + str(j) + '_cpu1b_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('tb1_' + str(j), 'tb1_' + str(j) + '_cpu1b_' + str(i))
            G.add_edge('tb1_' + str(j) + '_cpu1b_' + str(i), 'ws_cpu1b_' + str(i))

        for j in range(W_CPU4_CPU1b.shape[1]):
            w = 0.5 * W_CPU4_CPU1b[i][j]
            G.add_node('cpu4_' + str(j) + '_cpu1b_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'cpu4_' + str(j) + '_cpu1b_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('cpu4_' + str(j), 'cpu4_' + str(j) + '_cpu1b_' + str(i))
            G.add_edge('cpu4_' + str(j) + '_cpu1b_' + str(i), 'ws_cpu1b_' + str(i))

        for j in range(W_pontin_CPU1b.shape[1]):
            w = -0.5 * W_pontin_CPU1b[i][j]
            G.add_node('pontin_' + str(j) + '_cpu1b_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'pontin_' + str(j) + '_cpu1b_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('pontin_' + str(j), 'pontin_' + str(j) + '_cpu1b_' + str(i))
            G.add_edge('pontin_' + str(j) + '_cpu1b_' + str(i), 'ws_cpu1b_' + str(i))

    # motor
    for i in range(W_CPU1a_motor.shape[0]):
        G.add_node('motor_' + str(i),
                **{'class': 'Motor',
                    'name': 'motor_' + str(i) + '_s',
                    'init_motor': 0.,
                    'dummy': 0.
                    })

        G.add_node('ws_motor_' + str(i),
                **{'class': 'WeightedSum',
                    'name': 'ws_motor_' + str(i) + '_s',
                    'init_I': 0.,
                    'dummy': 0.
                    })

        G.add_edge('ws_motor_' + str(i), 'motor_' + str(i))

        for j in range(W_CPU1a_motor.shape[1]):
            w = W_CPU1a_motor[i][j]
            G.add_node('cpu1a_' + str(j) + '_motor_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'cpu1a_' + str(j) + '_motor_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('cpu1a_' + str(j), 'cpu1a_' + str(j) + '_motor_' + str(i))
            G.add_edge('cpu1a_' + str(j) + '_motor_' + str(i), 'ws_motor_' + str(i))

        for j in range(W_CPU1b_motor.shape[1]):
            w = W_CPU1b_motor[i][j]
            k = 0
            if j == 0:
                k = 1

            G.add_node('cpu1b_' + str(k) + '_motor_' + str(i),
                    **{'class': 'Synapse',
                        'name': 'cpu1b_' + str(k) + '_motor_' + str(i) + '_s',
                        'init_individual_input': 0.,
                        'w': float(w)
                        })

            G.add_edge('cpu1b_' + str(k), 'cpu1b_' + str(k) + '_motor_' + str(i))
            G.add_edge('cpu1b_' + str(k) + '_motor_' + str(i), 'ws_motor_' + str(i))
   
    #plt.figure(figsize=(18,18))
    #nx.draw_networkx(G,font_size=5,node_size=100,width=0.3)
    #plt.savefig('./network.jpg')

    return G


def create_lpu(file_name):
    """
    Create a generic LPU graph.

    Creates a GEXF file containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The GEXF
    file also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are either spiking or graded potential neurons;
    the Leaky Integrate-and-Fire model is used for the former, while the
    Morris-Lecar model is used for the latter (i.e., the neuron's membrane
    potential is deemed to be its output rather than the time when it emits an
    action potential). Synapses use either the alpha function model or a
    conductance-based model.

    Parameters
    ----------
    file_name : str
        Output GEXF file name.
    lpu_name : str
        Name of LPU. Used in port identifiers.
    N_sensory : int
        Number of sensory neurons.
    N_local : int
        Number of local neurons.
    N_proj : int
        Number of project neurons.

    Returns
    -------
    g : networkx.MultiDiGraph
        Generated graph.
    """

    g = create_lpu_graph()
    nx.write_gexf(g, file_name)


def create_input():

    uids_tl2 = ["tl2_" + str(i) for i in range(N_TL2)]
    uids_tn2 = ["tn2_" + str(i) for i in range(N_TN2)]
    uids = ["tl2_" + str(i) for i in range(N_TL2)]
    uids.extend(["tn2_" + str(i) for i in range(N_TN2)])
    uids = np.array(uids)
    uids_tl2 = np.array(uids_tl2)
    uids_tn2 = np.array(uids_tn2)
    Nt = 1500
    N_sensory_heading = N_TL2 + N_TN2
    N_sensory_vx = N_TN2
    N_sensory_vy = N_TN2

    h = np.load('h.npy')
    vx = np.load('vx.npy')
    vy = np.load('vy.npy')

    h_data = np.tile(h, (N_sensory_heading,1)).T
    vx_data = np.tile(vx, (N_sensory_vx,1)).T
    vy_data = np.tile(vy, (N_sensory_vy,1)).T


    with h5py.File('h.h5', 'w') as f:
        f.create_dataset('h/uids', data=uids)
        f.create_dataset('h/data', (Nt, N_sensory_heading),
                         dtype=np.float64,
                         data=h_data)

    with h5py.File('vx.h5', 'w') as f:
        f.create_dataset('v_x/uids', data=uids_tn2)
        f.create_dataset('v_x/data', (Nt, N_sensory_vx),
                         dtype=np.float64,
                         data=vx_data)

    with h5py.File('vy.h5', 'w') as f:
        f.create_dataset('v_y/uids', data=uids_tn2)
        f.create_dataset('v_y/data', (Nt, N_sensory_vy),
                         dtype=np.float64,
                         data=vy_data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='generic_path_integration.gexf.gz',
                        help='LPU file name')

    args = parser.parse_args()

    create_lpu(args.lpu_file_name)
    g = nx.read_gexf(args.lpu_file_name)
    create_input()

