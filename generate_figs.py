#!/usr/bin/env python

"""
@author: Rafael Menezes (github: r-menezes)
@date: 2025
@license: MIT License
"""

# Imports
import numpy as np

from figs_settings import *
from figs_code import fig2, fig3, fig4, fig5

import pandas as pd


# Auxiliary functions

## Data wrangling functions

def get_idx(df, var, idx):
    return np.sort([x for x in pd.unique(df[var]) if not np.isnan(x)])[idx]


def filtering(df, comp = None, disp = None, tau = None, hr = None, noise = None):
    res = np.full(len(df), True, dtype=bool)
    
    if comp is not None:
        if isinstance(comp, int):
            comp = get_idx(df, 'comp', comp)
        a = (df['comp'] == comp)
        res = res & a
    if disp is not None:
        if isinstance(disp, int):
            disp = get_idx(df, 'dispersal', disp)
        b = (df['dispersal'] == disp)
        res = res & b
    if tau is not None:
        if isinstance(tau, int):
            tau = get_idx(df, 'tau', tau)
        c = (df['tau'] == tau)
        res = res & c
    if hr is not None:
        if isinstance(hr, int):
            hr = get_idx(df, 'hr_stdev', hr)
        d = (df['hr_stdev'] == hr)
        res = res & d
    if noise is not None:
        if isinstance(noise, int):
            noise = get_idx(df, 'noise', noise)
        e = (df['noise'] == noise)
        res = res & e
    return df[res]


def load_dataframe(file_path):
    """
    Load a pandas dataframe from a file.
    """
    import os
    import pandas as pd

    file = os.path.basename(file_path)
    if file.endswith(".json"):
        new_df = pd.read_json(file_path)
    elif file.endswith(".parquet"):
        new_df = pd.read_parquet(file_path)
    elif file.endswith(".csv"):
        new_df = pd.read_csv(file_path)
    elif file.endswith(".pkl"):
        new_df = pd.read_pickle(file_path)
    elif file.endswith(".hdf"):
        new_df = pd.read_hdf(file_path)
    elif file.endswith(".feather"):
        new_df = pd.read_feather(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
        # warn the user if the file is not one of the supported types
        # print(f"Warning: {file} is not a supported file type.")
        # return None

    return new_df


def sci_round(n, float_precision=2):
    """
    Round a number to a given number of decimal places when written in scientific notation.
    """
    if n == 0:
        return 0
    elif np.isnan(n):
        return np.nan
    elif n == np.inf:
        return np.inf
    else:
        # calculate the number of decimal places to round to
        decimal_places = -int(np.floor(np.log10(abs(n)))) + float_precision - 1
        return round(n, decimal_places)


def df_data_wrangling(df):

    # movement
    # sigma_r^2 = tau*noise/2
    # 2.447746830680816 => radius of circle containing 95% of the probability mass of the 2D isotropic Gaussian
    df['noise'] = df['noise_amplitude']
    df['hr_stdev'] = np.sqrt(df['tau']*df['noise']/2)
    df['HRarea'] = np.pi*(2.447746830680816*df['hr_stdev'])**2

    # mean field
    mean_field_N = lambda birth, death, gamma, size: size*size*(birth - death)/gamma
    df['mfN'] = mean_field_N(df['birth_rate'], df['death_rate'], df['comp_rate'], df['env_size'])

    # carrying capacity
    df.loc[:, 'N*'] = df['final_N_org']/df['mfN']

    # predictions
    df.loc[:,'pred_n_hr'] = 1/df['hr_crowd']
    df.loc[:,'pred_n_pos'] = 1/df['pos_crowd']

    # short hand for names
    df.loc[:,'comp'] = df['comp_kernel'].astype(float)
    df.loc[:,'disp'] = df['dispersal'].astype(float)
    df.loc[:,'hr'] = df['HRarea'].astype(float)
    df.loc[:,'tau'] = df['tau'].astype(float)
    df.loc[:,'noise'] = df['noise'].astype(float)
    df.loc[:,'dispersal'] = df['dispersal'].astype(float)
    df.loc[:,'comp_kernel'] = df['comp_kernel'].astype(float)
    df.loc[:,'hr_stdev'] = df['hr_stdev'].astype(float)
    df.loc[:,'N*'] = df['N*'].astype(float)
    df.loc[:,'mov'] = df['mover_class'].astype(str)

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in df.columns:
        if col in numeric_columns:
            df[col] = df[col].map(lambda x: sci_round(x, float_precision=6))

    return df


# Main function

def main():
    import os
    import argparse
    from time import ctime, time

    parser = argparse.ArgumentParser(
                    prog='generate_figs.py',
                    description='This script generates the figures for Menezes et al. (2025).',
                    epilog='2025 CC-BY Rafael Menezes (github: r-menezes)')

    # input/output files
    parser.add_argument('fig2_dataset', type=str, nargs='?', help='Path of the dataset containing the simulations for the carrying capacity vs. home range size figure.', default='output/fig2_dataset.parquet')
    parser.add_argument('figs34_dataset', type=str, nargs='?', help='Path of the dataset containing the simulations for the heatmap and predicted vs simulated carrying capacity figures.', default='output/figs34_dataset.parquet')
    parser.add_argument('output', type=str, nargs='?', help='Path of the folder where to store the figures.', default='figs/')
    parser.add_argument('--tex', action='store_true', help='Use LaTeX for text rendering. Requires LaTeX to be installed on the system.')

    # parse arguments
    args = parser.parse_args()

    # If the user wants to use LaTeX for text rendering, set the rcParams accordingly
    if args.tex:
        mpl.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['CMU Serif', 'DejaVu Serif', 'Times New Roman', 'serif'],
            'font.sans-serif': ['CMU Sans Serif', 'DejaVu Sans', 'Arial', 'sans-serif'],
        })
    else:
        mpl.rcParams.update({
            'text.usetex': False,
            'font.family': 'sans-serif',
            'font.sans-serif': ['CMU Sans Serif', 'DejaVu Sans', 'Arial', 'sans-serif'],
            'font.serif': ['CMU Serif', 'DejaVu Serif', 'Times New Roman', 'serif'],
        })

    # check if the output folder exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load fig2 dataset
    now = ctime()
    print(f"{now} || Loading data for fig2...")
    t0 = time()
    treated_df_name = args.fig2_dataset.split('.')[0] + '_treated.parquet'
    if os.path.exists(treated_df_name):
        print(f"{now} || Treated dataset already exists. Loading {treated_df_name}...")
        df = pd.read_parquet(treated_df_name)
    else:
        print(f"{now} || Treated dataset not found. Loading {args.fig2_dataset}...")
        df = load_dataframe(args.fig2_dataset)
        df = df_data_wrangling(df)
        print(f"{now} || Saving treated dataset as {treated_df_name}...")
        df.to_parquet(treated_df_name)

    # fig2
    now = ctime()
    print(f"{now} || Generating fig2...")
    _ = fig2(df, save= args.output + 'fig2')

    # Load fig3 and fig4 dataset
    now = ctime()
    print(f"{now} || Loading data for fig3 and fig4...")

    treated_df_name = args.figs34_dataset.split('.')[0] + '_treated.parquet'
    if os.path.exists(treated_df_name):
        print(f"{now} || Treated dataset already exists. Loading {treated_df_name}...")
        df = pd.read_parquet(treated_df_name)
    else:
        print(f"{now} || Treated dataset not found. Loading {args.figs34_dataset}...")
        df = load_dataframe(args.figs34_dataset)
        df = df_data_wrangling(df)
        print(f"{now} || Saving treated dataset as {treated_df_name}...")
        df.to_parquet(treated_df_name)

    # fig3
    now = ctime()
    print(f"{now} || Generating fig3...")
    _ = fig3(df, save= args.output + 'fig3')

    # fig4
    now = ctime()
    print(f"{now} || Generating fig4...")
    _ = fig4(df, save= args.output + 'fig4')

    # fig4_no_restriction
    now = ctime()
    print(f"{now} || Generating fig4_no_restriction...")
    _ = fig4(df, save= args.output + 'fig4_no_restriction', filters=False)

    # fig5
    now = ctime()
    print(f"{now} || Generating fig5...")
    _ = fig5(save= args.output + 'fig5')

    tf = time()
    now = ctime()
    print(f"{now} || Done! Execution time: {tf-t0:.2f} seconds.")
    print(f"{now} || Figures saved in {args.output}.")

    return 0


if __name__ == "__main__":
    main()


