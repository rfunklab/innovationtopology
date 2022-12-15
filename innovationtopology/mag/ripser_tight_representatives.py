import os
import argparse
import subprocess
import pickle

from innovationtopology.ripser_tools import unpack_ripser_stdout
from innovationtopology.config import RIPSER_LOC, mag

subjects = mag['subjects']
levels = mag['levels']

score_min = mag['default_score_min']
edge_probability_threshold = mag['default_edge_probability_threshold']
dim = 2
close_holes = True
threshold = 1

# {close/open}_{score_min}_ep{edge_probability_threshold}_dim{gen_dim+1}_{subject}_{level}.csv
savestring = '{}_{}_ep{}_dim{}_{}_{}.csv'
rep_saveloc = '../data/temporal_holes-fields_of_study/ripser_tight_representatives'
spdist_saveloc = '../data/temporal_holes-fields_of_study/spdist'
# {score_min}_ep{edge_probability_threshold}_{subject}_{level}.loser_distance_matrix
spdist_savestring = '{}_ep{}_{}_{}.lower_distance_matrix'

rep_savestring = '{}_ep{}_dim{}_{}_{}_{}.pkl'
    
def run(score_min, levels=levels, subjects=subjects, threshold=threshold,
        spdist_saveloc=spdist_saveloc, rep_saveloc=rep_saveloc, edge_probability_threshold=edge_probability_threshold):

    for subject in subjects:

        for level in levels:

            print(f'running wos subject {subject}, {level}...')

            spdist_fname = spdist_savestring.format(score_min, edge_probability_threshold, subject, level)
            spdist_file = os.path.join(spdist_saveloc, spdist_fname)

            if not os.path.exists(spdist_file):
                print('skipping', spdist_file, ' not found')
                pass

            rc = [RIPSER_LOC, '--threshold', f'{threshold}', '--dim', f'{dim}', '--format', 'lower-distance', spdist_file]
            result = subprocess.run(rc, stdout=subprocess.PIPE)
            result_str = result.stdout.decode('utf-8')

            barcodes = []
            for d in range(dim+1):
                print(f'unpacking dimension {d}...')
                barcode = unpack_ripser_stdout(result_str, d)
                print('barcode', barcode)

                ofile = rep_savestring.format(score_min, edge_probability_threshold, d, subject, level, threshold)
                with open(os.path.join(rep_saveloc, ofile), 'wb') as f:
                    pickle.dump(barcode, f)

                barcodes.append(barcode)

            ofile = rep_savestring.format(score_min, edge_probability_threshold, 'all', subject, level, threshold)
            with open(os.path.join(rep_saveloc, ofile), 'wb') as f:
                pickle.dump(barcodes, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create diagram distance matrix')
    parser.add_argument('-nmin', '--nrecord-min', type=int, required=False, default=score_min,
                        help='minimum number of concept citations')
    parser.add_argument('-l', '--levels',  nargs='+',  required=False, default=levels,
                        help='list of levels to run')
    parser.add_argument('-s','--subjects', nargs='+',  required=False, default=subjects,
                        help='list of subjects to run')
    parser.add_argument('-ss', '--spdist-saveloc', type=str, required=False, default=spdist_saveloc,
                        help='location of sparse matrices to run')
    parser.add_argument('-rs', '--rep-saveloc', type=str, required=False, default=rep_saveloc,
                        help='location of sparse matrices to run')
    parser.add_argument('-t', '--threshold', type=float, required=False, default=threshold,
                        help='persistence threshold')
    parser.add_argument('-ep', '--edge-probability-threshold', type=float, required=False, default=edge_probability_threshold,
                        help='edge likelihood percentile threshold')

    args = parser.parse_args()
    run(args.nrecord_min, levels=args.levels, threshold=args.threshold,
        subjects=args.subjects, spdist_saveloc=args.spdist_saveloc, rep_saveloc=args.rep_saveloc, 
        edge_probability_threshold=args.edge_probability_threshold)
