import os
import time
import argparse
from datetime import datetime
import numpy as np
import networkx as nx
import pandas as pd
import gudhi as gd

from innovationtopology.db_tools import MAGDB
from innovationtopology.utils import save_lower_distance_matrix, save_nodes, threshold_edge_probability
from innovationtopology.config import mag

start = mag['start']
end = mag['end']

levels = mag['levels']
subjects = mag['subjects']

score_min = mag['default_score_min']
edge_probability_threshold = mag['default_edge_probability_threshold']
gen_dims = [0,1,2]
close_holes = True

# max homology dimension (compute up to betti_{})
max_dimension = 3

# {close/open}_{score_min}_ep{edge_probability_threshold}_dim{gen_dim+1}_{subject}_{level}.csv
savestring = '{}_{}_ep{}_dim{}_{}_{}.csv'
saveloc = '../data/temporal_holes-fields_of_study'
# {score_min}_{subject}_{level}.csv
betti_savestring = '{}_ep{}_{}_{}.csv'

def run(score_min, subjects, levels, save_for_ripser=False, edge_probability_threshold=edge_probability_threshold):

    prefix = 'close'

    # create networkx graph from knowledge area network within date range
    startdt = datetime.strptime(start, '%Y-%m-%d')
    enddt = datetime.strptime(end, '%Y-%m-%d')

    db = MAGDB()

    for subject in subjects:

        print(f'Beginning Subject: {subject}')

        db = MAGDB()
        parent_id = db.get_parent_id(subject)

        for level in levels:

            print(f'Beginning subject {subject}, level {level}')

            st = time.time()
            db = MAGDB()
            df = db.get_concept_network_by_level(parent_id, level, score_min)
            ed = time.time()
            print(f'took {ed - st} s')

            if df.shape[0] == 0:
                print('no papers for level:', level)
                continue

            print(df.head())
            # convert string date to datetime object
            # add seconds-since-unix-epoch column.
            # need to add time to get seconds, so assume each date is at min time (midnight?)
            df['seconds'] = df['Date'].apply((lambda x: datetime.combine(x, datetime.min.time()).timestamp()))
            mn = df['seconds'].min()
            # because anything before 1/1/1970 will have negative seconds,
            # add minimal seconds back to all rows to make all seconds positive
            df['seconds'] = df['seconds'] + np.abs(mn)+1
            print(df.shape)

            if df.shape[0] == 0:
                print('no papers for level:', level)
                continue

            # convert to relational form
            g = nx.Graph()
            df_fil = df[(df['Date'] > startdt.date()) & (df['Date'] < enddt.date())].sort_values('Date',ascending=False)

            if edge_probability_threshold > 0:
                nrecords_a = df_fil.groupby('FieldOfStudyA')['PaperId'].count().to_frame('concept_nrecord_ids_a')
                nrecords_b = df_fil.groupby('FieldOfStudyB')['PaperId'].count().to_frame('concept_nrecord_ids_b')
                df_fil = (df_fil.merge(nrecords_a, left_on='FieldOfStudyA', right_index=True)
                        .merge(nrecords_b, left_on='FieldOfStudyB', right_index=True))
                df_fil = threshold_edge_probability(df_fil, edge_probability_threshold, 
                                                    concept_a_col='FieldOfStudyA', concept_a_count_col='concept_nrecord_ids_a',
                                                    concept_b_col='FieldOfStudyB', concept_b_count_col='concept_nrecord_ids_b',
                                                    record_id_col='PaperId').sort_values('Date', ascending=False)

            mx = df_fil['seconds'].max()
            for idx, row in df_fil.iterrows():
                g.add_edge(row['FieldOfStudyA'],row['FieldOfStudyB'],weight=row['seconds']/mx)
            nodes = list(g.nodes())

            print('Number of Nodes: {}, Number of Edges: {}, Density: {}'.format(len(g.nodes()), len(g.edges()), len(g.nodes())/len(g.edges())))

            # convert to Gudhi list-of-list adjacency
            mat = nx.to_numpy_matrix(g)
            mat[mat == 0] = 1.1
            m = [mat[i,:i].tolist()[0] for i in range(mat.shape[0])]

            lower_distance_filename = betti_savestring.format(score_min, edge_probability_threshold, subject, level).replace('.csv','.lower_distance_matrix')
            save_lower_distance_matrix(mat, os.path.join(saveloc,'spdist',lower_distance_filename))

            node_saveloc = os.path.join(saveloc, 'nodes', betti_savestring.format(score_min, edge_probability_threshold, subject, level))
            save_nodes(nodes, node_saveloc)

            if save_for_ripser:
                # just save network data and move on
                continue
            
            print('Computing Rips Complex...')
            rips_complex = gd.RipsComplex(distance_matrix=m,max_edge_length=1.0)
            print('Constructing Simplex Tree...')
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

            start_time = time.time()
            print('Computing Persistence...')
            dgm = simplex_tree.persistence()
            end_time = time.time()
            print('Persistence took: {}s'.format(end_time - start_time))

            # save betti numbers to file
            betti_numbers = np.array(simplex_tree.betti_numbers(), dtype='int')
            np.savetxt(os.path.join(saveloc,'betti_numbers',betti_savestring.format(score_min, edge_probability_threshold, subject, level)),betti_numbers, delimiter=',')
            # save persistence diagram to file
            simplex_tree.write_persistence_diagram(os.path.join(saveloc,'diagrams',betti_savestring.format(score_min,edge_probability_threshold,subject,level)))

            gens = simplex_tree.flag_persistence_generators()

            for gen_dim in gen_dims:

                # are there non-trivial holes in this dimension
                if gen_dim <= len(gens[1])-1:
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

                    x[np.argsort(x[:, 4])]

                    # construct dataframe of hole-closing papers and their associated features
                    # construct dataframe of hole-closing papers and their associated features
                    res = []
                    sk1 = 'vd1'
                    sk2 = 'vd2'
                    dt = 'death_date'
                    for i in range(x.shape[0]):
                        if i % 100 == 0: print('Subject: {}, Level: {}, Dim: {}, {}/{}'.format(subject, level, gen_dim+1, i, x.shape[0]))
                        r = {}
                        r['vb1'] = nodes[int(x[i,0])]
                        r['vb2'] = nodes[int(x[i,1])]
                        r['vd1'] = nodes[int(x[i,2])]
                        r['vd2'] = nodes[int(x[i,3])]
                        r['lifetime'] = x[i,4]

                        r['birth_date'] = datetime.fromtimestamp(g[r['vb1']][r['vb2']]['weight']*mx - np.abs(mn)-1).date()
                        r['birth_point'] = g[r['vb1']][r['vb2']]['weight']
                        r['death_date'] = datetime.fromtimestamp(g[r['vd1']][r['vd2']]['weight']*mx - np.abs(mn)-1).date()
                        r['death_point'] = g[r['vd1']][r['vd2']]['weight']

                        es = [(e[0],e[1]) for e in g.edges(data=True) if e[2]['weight'] < r['death_point']]
                        h = g.edge_subgraph(es)
                        try:
                            r['shortest_path_before_death'] = nx.shortest_path_length(h, source=r['vd1'], target=r['vd2'])
                        except (nx.exception.NodeNotFound, nx.exception.NetworkXNoPath):
                            r['shortest_path_before_death'] = np.nan

                        rows = df[((df['FieldOfStudyA'] == r[sk1]) & (df['FieldOfStudyB'] == r[sk2])) | ((df['FieldOfStudyB'] == r[sk1]) & (df['FieldOfStudyA'] == r[sk2]))][['PaperId','Date','CitationCount']].copy() 

                        for k,v in r.items():
                            rows[k] = v
                        res.append(rows)

                    df_res = pd.concat(res, ignore_index=True)

                    filename = savestring.format(prefix, score_min, edge_probability_threshold, gen_dim+1, subject, level)
                    df_res.to_csv(os.path.join(saveloc,filename), index=False)

                else:
                    print('Gen Dim: {} trivial'.format(gen_dim))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute temporal holes for MAG')
    parser.add_argument('-sm', '--score-min', type=float, required=False, default=score_min,
                        help='minimum MAG score for both combined comcepts of paper')
    parser.add_argument('-s','--subjects', nargs='+', help='list of subjects_extended to run', required=False, default=subjects)
    parser.add_argument('-l','--levels', nargs='+', help='list of fields of study levels to run', required=False, default=levels)
    parser.add_argument('-r', '--save-for-ripser', action='store_true',
                    help='whether to save out a sparse distance matrix for ripser homology computation. \
                    will exit after writing file to disk, does not compute homology')
    parser.add_argument('-ep', '--edge-probability-threshold', type=float, required=False, default=edge_probability_threshold,
                        help='edge likelihood percentile threshold')

    args = parser.parse_args()
    levels = [int(l) for l in args.levels]
    run(args.score_min, args.subjects, levels, save_for_ripser=args.save_for_ripser, edge_probability_threshold=edge_probability_threshold)
