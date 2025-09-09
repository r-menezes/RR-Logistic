#!/usr/bin/env python

"""
@author: Rafael Menezes (github: r-menezes)
@date: 2025
@license: MIT License
"""

# Imports
import numpy as np

from figs_settings import *
from figs_code import fig2, fig3, fig4, fig_sup1, fig_sup2, fig_sup3, fig_sup4

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

def df_data_wrangling(df, hr_target_vals=None):

    # movement
    # sigma_r^2 = tau*noise/2
    # 2.447746830680816 => radius of circle containing 95% of the probability mass of the 2D isotropic Gaussian
    df['noise'] = df['noise_amplitude']
    # offset numerical errors from JSON limited float precision
    if hr_target_vals is not None:
        df['hr_stdev'] = hr_target_vals[np.abs(hr_target_vals[:, None] - np.sqrt(df['tau'] * df['noise'] * 0.5).values).argmin(axis=0)]
    else:
        df['hr_stdev'] = np.sqrt(df['tau'] * df['noise'] * 0.5)
    
    df['HRarea'] = np.pi*(2.447746830680816*df['hr_stdev'])**2

    # mean field
    mean_field_N = lambda birth, death, gamma, size: size*size*(birth - death)/(0.95*gamma) # correct for setting maximum competition distance
    df['mfN'] = mean_field_N(df['birth_rate'], df['death_rate'], df['comp_rate'], df['env_size'])

    # carrying capacity
    df.loc[:, 'N*'] = df['final_N_org']/df['mfN']
    df.loc[:, 'ave_N*'] = df['ave_N_org']/df['mfN']

    # predictions
    try:
        df.loc[:,'ave_pred_n_hr'] = 1/df['ave_hr_crowd']
    except KeyError:
        df.loc[:,'ave_pred_n_hr'] = None
    
    try:
        df.loc[:,'ave_pred_n_pos'] = 1/df['ave_pos_crowd']
    except KeyError:
        df.loc[:,'ave_pred_n_pos'] = None
    
    # predictions
    try:
        df.loc[:,'pred_n_hr'] = 1/df['final_hr_crowd']
    except KeyError:
        df.loc[:,'pred_n_hr'] = None
    
    try:
        df.loc[:,'pred_n_pos'] = 1/df['final_pos_crowd']
    except KeyError:
        df.loc[:,'pred_n_pos'] = None

    # handle dispersal if it is a string
    if df['dispersal'].dtype == 'object' or df['dispersal'].dtype == 'category':
        for idx, row in df.iterrows():
            if isinstance(row['dispersal'], str):
                try:
                    _disp_str = row['dispersal']
                    _disp_type, _disp_shape, _disp_scale = _disp_str.split('_')
                    df.at[idx, 'disp_type'] = str(_disp_type)
                    df.at[idx, 'disp_shape'] = float(_disp_shape)
                    df.at[idx, 'disp_scale'] = float(_disp_scale)
                    df.at[idx, 'dispersal'] = float(_disp_shape)
                except ValueError:
                    # if it cannot be converted to float, set it to NaN
                    df.at[idx, 'dispersal'] = np.nan

    # short hand for names
    df.loc[:,'comp'] = df['comp_kernel'].astype(float)
    df.loc[:,'dispersal'] = df['dispersal'].astype(float)
    df.loc[:,'disp'] = df['dispersal'].astype(float)
    df.loc[:,'hr'] = df['HRarea'].astype(float)
    df.loc[:,'tau'] = df['tau'].astype(float)
    df.loc[:,'noise'] = df['noise'].astype(float)
    df.loc[:,'comp_kernel'] = df['comp_kernel'].astype(float)
    df.loc[:,'hr_stdev'] = df['hr_stdev'].astype(float)
    df.loc[:,'N*'] = df['N*'].astype(float)
    df.loc[:,'mov'] = df['mover_class'].astype(str)

    numeric_columns = df.select_dtypes(include=[np.number]).columns

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
    
    # valid figures
    valid_figs = ['fig2', 'fig3', 'fig4', 'fig_sup1', 'fig_sup2', 'fig_sup3', 'fig_sup4']

    # input/output files
    parser.add_argument('-figs', '--figures', type=str, nargs='+', help=f'List of figures to generate. Options: {", ".join(valid_figs)}. A subset of figures preceded by "-" can be passed as argument, in this case all figures excpet the specified ones will be generated. Default: all.',
                       default=valid_figs)
    parser.add_argument('--fig2_dataset', type=str, nargs='?', help='Path of the dataset containing the simulations for the carrying capacity vs. home range size figure.', default='output/fig2_dataset.parquet')
    parser.add_argument('--gamma', type=str, help='Path of the dataset containing the simulations for the gamma distribution figure.', default='output/fig_sup2_dataset.parquet')
    parser.add_argument('--figs34_dataset', type=str, nargs='?', help='Path of the dataset containing the simulations for the heatmap and predicted vs simulated carrying capacity figures.', default='output/figs34_dataset.parquet')
    parser.add_argument('--fig_sup3_dataset', type=str, nargs='?', help='Path of the dataset containing the simulations for the supplementary figure 3.', default='output/fig_sup3_dataset.parquet')
    parser.add_argument('--fig_sup3_temporal', type=str, nargs='?', help='Path of the folder containing the temporal data of the simulations for the supplementary figure 3.', default='output/fig_sup3_temporal/')
    parser.add_argument('--fig_sup4_dataset', type=str, nargs='?', help='Path of the dataset containing the simulations showing that movement alone can shift carrying capacity above and below MF.', default='output/fig_sup4_dataset.parquet')
    parser.add_argument('-o', '--output', type=str, nargs='?', help='Path of the folder where to store the figures.', default='figs/')
    parser.add_argument('--tex', action='store_true', help='Use LaTeX for text rendering. Requires LaTeX to be installed on the system.')

    # parse arguments
    args = parser.parse_args()

    # If the user wants to use LaTeX for text rendering, set the rcParams accordingly
    if args.tex:
        mpl.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}',
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

    # HANDLING FIGURE SELECTION

    to_generate = []
    negative_specification = all([fig.startswith('-') for fig in args.figures])
    
    if negative_specification:
        print("Negative specification detected. Generating all figures except the specified ones.")
        skipped_figs = set()
        for fig in args.figures:
            if fig.startswith('-'):
                fig = fig[1:]
                if fig in valid_figs:
                    skipped_figs.add(fig)
                    print(f"Skipping figure: {fig}")
                else:
                    print(f"Warning: {fig} is not a valid figure. Skipping...")
            else:
                if fig not in valid_figs:
                    print(f"Invalid figure: {fig}. Valid figures are: {', '.join(valid_figs)}. Skipping...")
        
        to_generate = [fig for fig in valid_figs if fig not in skipped_figs]
        print(f"Figures to be generated: {', '.join(to_generate)}")
    
    for fig in args.figures:
        if fig.startswith('-'):
            print(f"Some, but not all, figures were specified with a negative sign. Please specify the figures to generate without the negative sign. Skipping {fig}.")
            continue
        if fig in valid_figs:
            to_generate.append(fig)
        else:
            print(f"Invalid figure: {fig}. Valid figures are: {', '.join(valid_figs)}. Skipping...")
    if len(to_generate) == 0:
        print("No valid figures specified. Exiting...")
        return 1
    print(f"Figures to be generated: {', '.join(to_generate)}")

    #  FIGURE GENERATION STARTS HERE
    
    t0 = time()

    #  Dataset handling

    # Load fig2 dataset
    if 'fig2' in to_generate:
        now = ctime()
        print(f"{now} || Loading data for fig2...")
        treated_df_name = args.fig2_dataset.split('.')[0] + '_treated.parquet'
        if os.path.exists(treated_df_name):
            print(f"{now} || Treated dataset already exists. Loading {treated_df_name}...")
            df2 = pd.read_parquet(treated_df_name)
        else:
            print(f"{now} || Treated dataset not found. Loading {args.fig2_dataset}...")
            df2 = load_dataframe(args.fig2_dataset)
            df2 = df_data_wrangling(df2)
            print(f"{now} || Saving treated dataset as {treated_df_name}...")
            df2.to_parquet(treated_df_name)

    # Load fig3 and fig4 dataset
    if any(fig in to_generate for fig in ['fig3', 'fig4']):
        now = ctime()
        print(f"{now} || A figure that requires the figs34_dataset was marked for generation.")
        print(f"{now} || Loading data for fig3 and fig4...")

        treated_df_name = args.figs34_dataset.split('.')[0] + '_treated.parquet'
        if os.path.exists(treated_df_name):
            print(f"{now} || Treated dataset already exists. Loading {treated_df_name}...")
            df34 = pd.read_parquet(treated_df_name)
        else:
            print(f"{now} || Treated dataset not found. Loading {args.figs34_dataset}...")
            df34 = load_dataframe(args.figs34_dataset)
            _hr_target_vals = np.array([0.001     , 0.00115478, 0.00133352, 0.00153993, 0.00177828, 0.00205353, 0.00237137, 0.00273842, 0.00316228, 0.00365174, 0.00421697, 0.00486968, 0.00562341, 0.00649382, 0.00749894, 0.00865964, 0.01      , 0.01154782, 0.01333521, 0.01539927, 0.01778279, 0.02053525, 0.02371374, 0.0273842 , 0.03162278, 0.03651741, 0.04216965, 0.04869675, 0.05623413, 0.06493816, 0.07498942, 0.08659643, 0.1       , 0.1154782 , 0.13335214, 0.15399265, 0.17782794, 0.2053525 , 0.23713737, 0.27384196, 0.31622777, 0.36517413, 0.4216965 , 0.48696753, 0.56234133, 0.64938163, 0.74989421, 0.86596432, 1.0        ])
            df34 = df_data_wrangling(df34, hr_target_vals=_hr_target_vals)
            print(f"{now} || Saving treated dataset as {treated_df_name}...")
            df34.to_parquet(treated_df_name)

    # Load fig_sup3 dataset
    if 'fig_sup3' in to_generate:
        now = ctime()
        print(f"{now} || Loading data for fig_sup3...")
        treated_df_name = args.fig_sup3_dataset.split('.')[0] + '_treated.parquet'
        if os.path.exists(treated_df_name):
            print(f"{now} || Treated dataset already exists. Loading {treated_df_name}...")
            df_sup3 = pd.read_parquet(treated_df_name)
        else:
            print(f"{now} || Treated dataset not found. Loading {args.fig_sup3_dataset}...")
            df_sup3 = load_dataframe(args.fig_sup3_dataset)
            df_sup3 = df_data_wrangling(df_sup3)
            print(f"{now} || Saving treated dataset as {treated_df_name}...")
            df_sup3.to_parquet(treated_df_name)

    # Figure generation

    if 'fig2' in to_generate:
        # fig2
        now = ctime()
        print(f"{now} || Generating fig2...")
        _ = fig2(df2, save= args.output + 'fig2')

    # fig3
    if 'fig3' in to_generate:
        now = ctime()
        print(f"{now} || Generating fig3...")
        _ = fig3(df34, save= args.output + 'fig3')

    # fig4
    if 'fig4' in to_generate:
        now = ctime()
        print(f"{now} || Generating fig4...")
        _ = fig4(df34, save= args.output + 'fig4')

    # fig_sup1
    if 'fig_sup1' in to_generate:
        now = ctime()
        print(f"{now} || Generating fig_sup1...")
        _ = fig_sup1(save= args.output + 'fig_sup1')

    # fig_sup2
    if 'fig_sup2' in to_generate:
        # Load fig_sup2 dataset
        now = ctime()
        print(f"{now} || Checking if gamma argument is provided...")
        if args.gamma:
            cleaned_gamma_df_name = args.gamma.split('.')[0] + '_treated.parquet'
            gamma_df_name = args.gamma
            if not os.path.exists(gamma_df_name) and not os.path.exists(cleaned_gamma_df_name):
                print(f"{now} || No dataset provided for fig_sup2 (gamma). Skipping...")
            else:
                print(f"{now} || Loading data for fig_sup2 (gamma)...")
                if os.path.exists(cleaned_gamma_df_name):
                    print(f"{now} || Treated dataset already exists. Loading {cleaned_gamma_df_name}...")
                    df = pd.read_parquet(cleaned_gamma_df_name)
                else:
                    print(f"{now} || Treated dataset not found. Loading {gamma_df_name}...")
                    df = load_dataframe(gamma_df_name)
                    df = df_data_wrangling(df)
                    print(f"{now} || Saving treated dataset as {cleaned_gamma_df_name}...")
                    df.to_parquet(cleaned_gamma_df_name)

                # fig_sup2
                now = ctime()
                print(f"{now} || Generating fig_sup2...")
                fig_sup2(df, save= args.output + 'fig_sup2')
        else:
            print(f"{now} || fig_sup2 was marked for generation, but no gamma dataset was provided. Skipping...")

    # fig_sup3
    if 'fig_sup3' in to_generate:
        # fig_sup3
        now = ctime()
        print(f"{now} || Data handling for fig_sup3...")
        # order = ['SS', 'OU', 'BM']
        hrs_idxs = [0, 7, -1]
        df_ou = df_sup3[df_sup3['mover_class'] == 'OU'].copy()
        df_ou1 = filtering(df_ou, hr=hrs_idxs[0]).copy()
        df_ou1.loc[:, 'order'] = 1
        df_ou2 = filtering(df_ou, hr=hrs_idxs[1]).copy()
        df_ou2.loc[:, 'order'] = 2
        df_ou3 = filtering(df_ou, hr=hrs_idxs[2]).copy()
        df_ou3.loc[:, 'order'] = 3
        df_ou = pd.concat([df_ou1, df_ou2, df_ou3], ignore_index=True)
        df_sup3 = pd.concat([df_ou], ignore_index=True)

        print(f"{now} || Generating fig_sup3...")
        _ = fig_sup3(df=df_sup3,
                    folder_address=args.fig_sup3_temporal,
                    colvar='order',
                    rowvar='disp',
                    save= args.output + 'fig_sup3')

    # fig_sup4
    if 'fig_sup4' in to_generate:
        now = ctime()
        print(f"{now} || Note: fig_sup4 is essentially the same as fig2, but with different data.")
        print(f"{now} || Loading data for fig_sup4...")
        if args.fig_sup4_dataset:
            treated_df_name = args.fig_sup4_dataset.split('.')[0] + '_treated.parquet'
            if os.path.exists(treated_df_name):
                now = ctime()
                print(f"{now} || Treated dataset already exists. Loading {treated_df_name}...")
                df_sup4 = pd.read_parquet(treated_df_name)
            else:
                now = ctime()
                print(f"{now} || Treated dataset not found. Loading {args.fig_sup4_dataset}...")
                df_sup4 = load_dataframe(args.fig_sup4_dataset)
                df_sup4 = df_data_wrangling(df_sup4)
                now = ctime()
                print(f"{now} || Saving treated dataset as {treated_df_name}...")
                df_sup4.to_parquet(treated_df_name)
        else:
            now = ctime()
            print(f"{now} || No dataset provided for fig_sup4. Skipping...")
            return 1
        now = ctime()
        print(f"{now} || Generating fig_sup4...")
        _ = fig_sup4(df_sup4, save= args.output + 'fig_sup4')

    tf = time()
    now = ctime()
    print(f"{now} || Done! Execution time: {tf-t0:.2f} seconds.")
    print(f"{now} || Figures saved in {args.output}.")

    return 0


if __name__ == "__main__":
    main()


