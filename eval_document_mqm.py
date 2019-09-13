import argparse
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

"""
Script to evaluate document-level MQM as in the WMT19 shared task.
"""


def read_scores(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            doc_id, score = line.strip().split()
            data[doc_id] = score

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', help='System file')
    parser.add_argument('gold', help='Gold output file')
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help='Show all metrics (Pearson r, Spearman r, MAE, '
                             'RMSE). By default, it only computes Pearson r.')
    parser.add_argument('--tsv', action='store_true',
                        help='Save results in TSV format.')
    args = parser.parse_args()

    system_dict = read_scores(args.system)
    gold_dict = read_scores(args.gold)
    # assert len(system_dict) == len(gold_dict), \
    #     'Number of gold and system values differ: system {} gold {}'.format(
    #         len(system_dict), len(gold_dict))
    # assert set(system_dict.keys()) == set(gold_dict.keys()):
    #     'gold and system document ids differ: {} not in both'.format(
    #         set(system_dict.keys()).symmetric_difference(set(gold_dict.keys())))
    if set(system_dict.keys()) != set(gold_dict.keys()):
        print('warning: gold and system document ids differ: {} not in both '
              '(defaulting to an MQM of 100 for these documents)'.format(
                  set(system_dict.keys()).symmetric_difference(set(gold_dict.keys()))),
            file=sys.stderr)
        system_dict = defaultdict(lambda: 100, system_dict)
        gold_dict = defaultdict(lambda: 100, gold_dict)

    # get the scores in the same order
    doc_ids = list(gold_dict.keys())
    gold_scores = np.array([float(gold_dict[doc_id]) for doc_id in doc_ids])
    sys_scores = np.array([float(system_dict[doc_id]) for doc_id in doc_ids])

    # pearsonr and spearmanr return (correlation, p_value)
    pearson = pearsonr(gold_scores, sys_scores)[0]
    if args.tsv:
        header = []
        values = []
        header.append('mqm_pearson')
        values.append(str(round(pearson, 4)))

        if args.verbose:
            spearman = spearmanr(gold_scores, sys_scores)[0]

            diff = gold_scores - sys_scores
            mae = np.abs(diff).mean()
            rmse = (diff ** 2).mean() ** 0.5

            header.extend(['mqm_spearman', 'mqm_mae', 'mqm_rmse'])
            values.extend([str(round(spearman, 4)), str(round(mae, 4)), str(round(rmse, 4))])

            print('\t'.join(header))
            print('\t'.join(values))
    else:
        print('Pearson correlation: %.4f' % pearson)

        if args.verbose:
            spearman = spearmanr(gold_scores, sys_scores)[0]

            diff = gold_scores - sys_scores
            mae = np.abs(diff).mean()
            rmse = (diff ** 2).mean() ** 0.5

            print('Spearman correlation: %.4f' % spearman)
            print('MAE: %.4f' % mae)
            print('RMSE: %.4f' % rmse)
