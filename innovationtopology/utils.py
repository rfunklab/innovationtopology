import csv
import pickle
from typing import List, Callable
from multiprocessing.pool import Pool
from datetime import datetime
from functools import partial

import networkx as nx
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
            writer.writerow(mat[i,:i].tolist()[0])

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
            if not isinstance(k, int):
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
        if isinstance(rep[1][0], dict):
            network_dict[rep[0]] = create_hole_network(rep[1][0], dim=dim, rename=nodes)
        else:
            network_dict[rep[0]] = nx.Graph()

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
    network_df['death_point'] = network_df['point'].apply(lambda x: x[1] if len(x) == 2 else np.inf)
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

def aps_date_transform(mn, mx, w):
    return datetime.fromtimestamp(w*mx - np.abs(mn)-1).date()

def mag_date_transform(mn, mx, w):
    return datetime.fromtimestamp(w*mx - np.abs(mn)-1).date()

def wos_date_transform(w):
    return int(w)

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

def create_report_dict(x: list, g: nx.Graph, nodes: list, 
                    date_transform: Callable, df: pd.DataFrame, df_all: pd.DataFrame, 
                    concept_a_col: str, concept_b_col: str,                        
                    record_id_col: str, i: int) -> dict:
    if i % 100 == 0: print('{}/{}'.format(i,x.shape[0]))
    r = {}
    r['vb1'] = nodes[int(x[i,0])]
    r['vb2'] = nodes[int(x[i,1])]
    r['vd1'] = nodes[int(x[i,2])]
    r['vd2'] = nodes[int(x[i,3])]
    r['lifetime'] = x[i,4]

    r['birth_date'] = date_transform(g[r['vb1']][r['vb2']]['weight'])
    r['birth_point'] = g[r['vb1']][r['vb2']]['weight']
    r['death_date'] = date_transform(g[r['vd1']][r['vd2']]['weight'])
    r['death_point'] = g[r['vd1']][r['vd2']]['weight']

    es = [(e[0],e[1]) for e in g.edges(data=True) if e[2]['weight'] < r['death_point']]
    h = g.edge_subgraph(es)
    try:
        r['shortest_path_before_death'] = nx.shortest_path_length(h, source=r['vd1'], target=r['vd2'])
    except (nx.exception.NodeNotFound, nx.exception.NetworkXNoPath): 
        r['shortest_path_before_death'] = np.nan

    rids = df[((df[concept_a_col] == r['vd1']) & (df[concept_b_col] == r['vd2'])) | ((df[concept_b_col] == r['vd1']) & (df[concept_a_col] == r['vd2']))][record_id_col]
    rows = df_all[df_all[record_id_col].isin(list(rids))].copy()

    for k,v in r.items():
        rows[k] = v
    return rows

def unpack_gudhi_generators(gens: list, gen_dim: int, g: nx.Graph, nodes: list, 
                        date_transform: Callable, df: pd.DataFrame, df_all: pd.DataFrame,
                        concept_a_col: str = 'concept_a', concept_b_col: str = 'concept_b',                        
                        record_id_col: str = 'record_id',
                        ncores=1) -> pd.DataFrame:

    # are there non-trivial holes in this dimension
    if gen_dim > len(gens[1])-1:
        print('Gen Dim: {} trivial'.format(gen_dim))
        return pd.DataFrame()
    # create array of node indices and their lifetimes
    # dimensions are as follows:
    # 0: first birth vertex index
    # 1: second birth vertex index
    # 2: first death vertex index
    # 3: second death vertex index
    # 4: lifetime of hole
    x = np.empty((gens[1][gen_dim].shape[0],5))
    for i in range(gens[1][gen_dim].shape[0]):
        r = gens[1][gen_dim][i]
        x[i,:4] = r[:4]
        b = g[nodes[r[0]]][nodes[r[1]]]['weight']
        d = g[nodes[r[2]]][nodes[r[3]]]['weight']
        x[i,4] = d - b

    x[np.argsort(x[:,4])]

    # construct dataframe of hole-closing papers and their associated features
    res = []
    if ncores > 1:
        with Pool(ncores) as pool:
            f = partial(create_report_dict, x, g, nodes, 
                        date_transform, df, df_all, 
                        concept_a_col, concept_b_col, record_id_col)
            res = pool.map(f, range(x.shape[0]))
    else:
        for i in range(x.shape[0]):
            rows = create_report_dict(i, x, g, nodes, date_transform, df, df_all, 
                                    concept_a_col=concept_a_col, concept_b_col=concept_b_col, 
                                    record_id_col=record_id_col)
            res.append(rows)

    df_res = pd.concat(res, ignore_index=True)
    return df_res


