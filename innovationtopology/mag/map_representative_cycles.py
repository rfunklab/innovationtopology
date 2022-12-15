import os
import pandas as pd
import csv
import numpy as np
import networkx as nx

from innovationtopology.db_tools import MAGDB
from innovationtopology.utils import create_network_df, fuzzy_merge
from innovationtopology.config import mag

levels = mag['levels']
subjects = mag['subjects']

dim = 1
score_min = mag['default_score_min']
edge_probability_threshold = mag['default_edge_probability_threshold']
lifetime_cutoff = 1

dataloc = '../data/temporal_holes-fields_of_study'

def map_reps_to_pairs(subject, level, edge_probability_threshold):

    filestring = f'{score_min}_ep{edge_probability_threshold}_dim{dim}_{subject}_{level}_{lifetime_cutoff}.csv'
    close_filestring = f'close_{score_min}_ep{edge_probability_threshold}_dim{dim}_{subject}_{level}.csv'
    saveloc = os.path.join(dataloc, 'ripser_tight_representatives', 'joined', filestring)

    rep_loc = os.path.join(dataloc, 'ripser_tight_representatives', filestring.replace('csv','pkl'))
    node_loc = os.path.join(dataloc, 'nodes', f'{score_min}_ep{edge_probability_threshold}_{subject}_{level}.csv')

    ldm_fname = os.path.join(dataloc, 'spdist', f'{score_min}_ep{edge_probability_threshold}_{subject}_{level}.lower_distance_matrix')
    with open(ldm_fname, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        nrows = sum(1 for row in reader) + 1
    ldm = np.zeros((nrows,nrows))
    with open(ldm_fname, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, line in enumerate(reader):
            ldm[i+1,:i+1] = line
    ldm = ldm + ldm.T

    nodes = pd.read_csv(node_loc, header=None)
    nodes = nodes['0'].to_dict()

    # connect to mysql server and load the entire dataset (will be used as null distribution)
    db = MAGDB()

    # get field of study normalized names
    fos_dict = db.get_field_of_study_names()

    # now rename nodes to their normalized name
    nodes = {k:fos_dict[v] for k,v in nodes.items()}

    network_df = create_network_df(rep_loc, nodes, dim)

    dtstr = 'Date'

    closure_df = pd.read_csv(os.path.join(dataloc, close_filestring))
    closure_df = closure_df.sort_values(by=dtstr).drop_duplicates(subset=['vb1','vb2','vd1','vd2'])

    N = 1e6
    closure_df['birth_point'] = (closure_df['birth_point']*N).round(6).astype(int) 
    network_df['birth_point'] = (network_df['birth_point']*N).round(6).astype(int)

    closure_df['death_point'] = (closure_df['death_point']*N).round(6).astype(int) 
    network_df['death_point'] = (network_df['death_point']*N).round(6).astype(int)

    # full_df = closure_df.merge(network_df, on='lifetime', how='inner')
    full_df = fuzzy_merge(closure_df, network_df)
    # full_df = closure_df.merge(network_df, on=['birth_point', 'death_point'], how='inner')
    full_df['birth_point'] = full_df['birth_point'] / N
    full_df['death_point'] = full_df['death_point'] / N

    # rename birth and death concept pairs
    to_rename = ['vd1','vd2','vb1','vb2']
    for tr in to_rename:
        full_df[tr] = full_df[tr].apply(lambda x: fos_dict[int(x)] if not np.isnan(x) else None)

    print('closure_df', closure_df.shape)
    print('network_df', network_df.shape)
    print('full_df', full_df.shape)

    full_df['full_overlap'] = full_df.apply(lambda x: x['vd1'] in x['graph'] and x['vd2'] in x['graph'], axis=1)
    full_df['partial_overlap'] = full_df.apply(lambda x: x['vd1'] in x['graph'] or x['vd2'] in x['graph'], axis=1)

    print('full overlap', full_df[full_df['full_overlap']].shape[0] / full_df.shape[0])
    print('partial overlap', full_df[full_df['partial_overlap']].shape[0] / full_df.shape[0])
    print('expected overlap', 1 - (full_df.shape[0] - closure_df.shape[0]) / full_df.shape[0])

    full_df.to_pickle(saveloc.replace('.csv','.pkl'))
    full_df_csv = full_df.copy()
    full_df_csv['graph'] = full_df_csv['graph'].apply(lambda x: list(x.edges(data=True)) if isinstance(x, nx.Graph) else x)
    full_df_csv.to_csv(saveloc)

if __name__ == "__main__":

    for subject in subjects:
        for level in levels:
            print(f'running subject {subject}, level {level}')
            map_reps_to_pairs(subject, level, edge_probability_threshold=edge_probability_threshold)
