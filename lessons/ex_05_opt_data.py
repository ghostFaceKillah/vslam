import itertools

import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('results_diff_errors.csv')
    df = df.iloc[:, 1:]

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df.groupby(['max_px_distance', 'max_hamming_distance']).euclidean_err.mean().unstack()

    df = df[df.max_allowed_error < 0.5]

    cols = ['max_px_distance', 'max_hamming_distance', 'minimum_number_of_matches','max_allowed_error']

    for twocols in itertools.combinations(cols, 2):
        print(80 * '-')
        print(df.groupby(list(twocols)).euclidean_err.mean().unstack().round(2))



    x = 1