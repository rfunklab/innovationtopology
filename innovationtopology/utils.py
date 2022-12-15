import csv
import pickle
import networkx as nx
from typing import List

import pandas as pd
import numpy as np

def save_nodes(nodelist: List, saveloc: str) -> None:
    nodes = pd.Series(nodelist)
    nodes.to_csv(saveloc, index=False)

def save_lower_distance_matrix(mat: np.array, saveloc: str) -> None:
    np.fill_diagonal(mat, 0)
    with open(saveloc, 'w') as csvf:
        writer = csv.writer(csvf)
        for i in range(1,mat.shape[0]):
            writer.writerow(mat[i,:i])

def read_representatives(filename: str) -> dict:
    with open(filename, 'rb') as r_obj:
        return pickle.load(r_obj)

def create_1d_hole_network(d: dict) -> nx.Graph:
    G = nx.Graph()
    for k,v in d.items():
        G.add_edge(*k, weight=v)
    return G

def create_2d_hole_network(d: dict) -> nx.Graph:
    print()

def create_hole_network(d: dict, dim: int = 1, rename: dict = None) -> nx.Graph:
    if rename is not None:
        old_d = d.copy()
        d = {}
        for k,v in old_d.items():
            nk = (rename[n] for n in k)
            d[nk] = v

    if dim == 1:
        return create_1d_hole_network(d)
    if dim == 2:
        return create_2d_hole_network(d)
    else:
        raise ValueError(f'drawing in dimension {dim} unsupported')

def create_network_df(rep_loc: str, nodes: dict, dim: int) -> pd.DataFrame:
    reps = read_representatives(rep_loc)

    network_dict = {}
    for rep in reps:
        network_dict[rep[0]] = create_hole_network(rep[1][0], dim=dim, rename=nodes)

    network_df = pd.DataFrame([list(network_dict.keys()), list(network_dict.values())]).T
    network_df.columns = ['point', 'graph']

    to_add = []
    n_multi = 0
    for idx, row in network_df.iterrows():
        ccs = list(nx.connected_components(row['graph']))
        cc = len(ccs)
        if cc > 1:
            n_multi += cc
            for cc_g in ccs:
                ta = row.copy()
                ta['graph'] = nx.subgraph(row['graph'], cc_g).copy()
                to_add.append(ta.to_frame().T)

    network_df = pd.concat([network_df] + to_add, axis=0, ignore_index=True)
    network_df['birth_point'] = network_df['point'].apply(lambda x: x[0])
    network_df['death_point'] = network_df['point'].apply(lambda x: x[1])
    network_df['lifetime'] = network_df['death_point'] - network_df['birth_point']

    return network_df

def fuzzy_merge(closure_df: pd.DataFrame, network_df: pd.DataFrame, on: List = ['birth_point','death_point']) -> pd.DataFrame:
    # lifetimes expected in integers
    full_df = closure_df.merge(network_df, on=on, how='inner')
    adfs = []
    for col in on:
        fp1 = network_df.copy()
        fp1[col] += 1
        adfs.append(closure_df.merge(fp1, on=on, how='inner'))

        fm1 = network_df.copy()
        fm1[col] -= 1
        adfs.append(closure_df.merge(fm1, on=on, how='inner'))
    return pd.concat([full_df] + adfs, axis=0, ignore_index=True)

def compute_nrecords(nrecord_pct, num_records):
    return int(np.around((nrecord_pct/100.)*num_records))

def threshold_edge_probability(df_fil: pd.DataFrame, edge_probability_threshold: float, 
                            concept_a_col: str = 'concept_a', concept_a_count_col: str = 'concept_nrecord_ids_a',
                            concept_b_col: str = 'concept_b', concept_b_count_col: str = 'concept_nrecord_ids_b',
                            record_id_col: str = 'record_id') -> pd.DataFrame:
    concept_count_df = (pd.concat([df_fil[[concept_a_col, concept_a_count_col]], df_fil[[concept_b_col, concept_b_count_col]]
                                .rename({concept_b_col:concept_a_col,concept_b_count_col:concept_a_count_col}, axis=1)], ignore_index=True)
                                .groupby(concept_a_col).max())
    node_occur_prob = concept_count_df / concept_count_df.sum()
    df_pct = df_fil.merge(node_occur_prob.rename({concept_a_count_col:'concept_probability_a'}, axis=1), left_on=concept_a_col, right_index=True)
    df_pct = df_pct.merge(node_occur_prob.rename({concept_a_count_col:'concept_probability_b'}, axis=1), left_on=concept_b_col, right_index=True)
    paper_count = df_fil[record_id_col].nunique()
    edge_count_df = (df_fil.groupby([concept_a_col,concept_b_col])[record_id_col]
                    .count()
                    .to_frame(record_id_col)
                    .rename({record_id_col:'edge_probability'}, axis=1))
    edge_occur_prob = edge_count_df / paper_count
    
    df_pct = df_pct.merge(edge_occur_prob, left_on=[concept_a_col,concept_b_col], right_index=True)

    df_pct['prob_edge_weight'] = np.log10(df_pct['edge_probability'] / (df_pct['concept_probability_a'] * df_pct['concept_probability_b']))
    pctile_val = df_pct['prob_edge_weight'].quantile(edge_probability_threshold)
    return df_fil.loc[df_pct['prob_edge_weight'] >= pctile_val]