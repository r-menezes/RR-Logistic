# Range-Resident Logistic Model


[![DOI](https://zenodo.org/badge/975705705.svg)](https://doi.org/10.5281/zenodo.15312822)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.02.09.637279-blue)](https://doi.org/10.1101/2025.02.09.637279)


## Authorship

**Authors:**
- Rafael Menezes (maintainer) $^{1,2,3}$
    - github: [r-menezes](https://github.com/r-menezes)
    - email: r.menezes at ictp-saifr dot org 
- Justin M. Calabrese $^{2,4,5}$
- William F. Fagan $^{5}$
- Paulo Inácio Prado $^{3}$
- Ricardo Martinez-Garcia $^{2,1}$

$^{1}$ ICTP South American Institute for Fundamental Research and Instituto de Física Teórica, Universidade Estadual Paulista - UNESP, São Paulo, Brazil.
$^{2}$ Center for Advanced Systems Understanding (CASUS), Helmholtz-Zentrum Dresden-Rossendorf (HZDR), Görlitz, Germany.
$^{3}$ Department of Ecology, Institute of Biosciences, University of São Paulo, São Paulo, Brazil.
$^{4}$ Department of Ecological Modelling, Helmholtz Centre for Environmental Research – UFZ, Leipzig, Germany.
$^{5}$ Department of Biology, University of Maryland, College Park, MD, USA.

---

This repository contains the implementation of the range-resident logistic model, algonside helper code to run the simulations and reproduce the analyses presented in the accompanying paper.

## Setup

### Clone the repository:

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/r-menezes/rr-logistic.git
```

### Install dependencies:

#### Option 1: Conda installation

We suggest the use of [mamba](https://github.com/mamba-org/mamba) to manage dependencies. The instructions below are identical if using `conda` instead of `mamba`. To set up the environment:

1.  **Install mamba:** If you don't have conda installed, follow the instructions on the [official mamba website](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

2.  **Create the environment:** Navigate to the repository's root directory in your terminal and run:
    
    ```bash
    conda env create -f environment.yml
    ```

    This command reads the `environment.yml` file and installs all the necessary packages (including Python 3.12, numpy, pandas, scipy, matplotlib, seaborn, palettable, and pyarrow) into a new conda environment named `rr-logistic`.

3.  **Activate the environment:** Before running any scripts, activate the newly created environment:
    ```bash
    conda activate rr-logistic
    ```

4. **Install additional packages:** The package `p_tqdm` has to be installed using `pip`. To install it, run:
    ```bash
    pip install p_tqdm
    ```

Now you have all the required packages installed and can proceed with running the simulations and analyses.

#### Option 2: Pip installation

Alternatively, you can install the required packages using [pip]((https://pip.pypa.io/en/stable/installation/)). Before running the following command, ensure you have Python 3.12 installed on your system, as well as pip. We recommend using a virtual environment to avoid conflicts with other packages.

```bash
pip install -r requirements.txt
```

This command reads the `requirements.txt` file and installs all the necessary packages (numpy, pandas, scipy, matplotlib, seaborn, palettable, p_tqdm, and pyarrow).

## Reproducing the Figures

To reproduce the figures presented in the paper, simply run the `generate_figs.py` script. This script generates the figures based on the `fig2_dataset.parquet` and `figs34_dataset.parquet` files, which contain the aggregated results from the simulations.

```bash
python generate_figs.py
```

The script will create the figures in the `figs/` directory. The figures are saved in PNG format.

To use different datasets or customize the output directory, you can specify the paths as command-line arguments. The script's help message can be accessed with the `-h` or `--help` flag. By default, the script avoids using LaTeX for rendering the figures. If you want to use LaTeX, you can set the `--tex` flag when running the script. The usage is as follows:

```bash
python generate_figs.py [-h] [--tex] [fig2_dataset] [figs34_dataset] [output]
```

Note that the supplementary figures order is not the same as in the paper.


## Reproducing the Analyses

Follow these steps to reproduce the results:

### Define Experiments:

Create a JSON file (e.g., `experiments.json`) specifying the parameter combinations for your simulations. This will be used by the `experiment_planner` function (part of `run_experiments.py`) to create a full factorial exploration of the parameter space and generate individual parameter files for each simulation run. 

The documentation for the `experiment_planner` is available in the source code. The structure of `experiments.json` should look like this:

```json
{
    "parameter_levels": {
        "tau": [1.0e-06, "..." , 1.0e+00],
        "noise": [],
        "hr_stdev": [0.001, "..." , 1.0],
        "dispersal": [1.0e-03, 1.0e-02, 1.0e-01],
        "comp_kernel": [1.0e-03, 1.0e-02, 1.0e-01],
        "env_size": [1.0],
        "birth_rate": [1.1],
        "death_rate": [0.1],
        "comp_rate": [0.002],
        "steps": [100000],
        "burn_in_steps": [50000],
        "data_interval": [500],
        "max_abundance": [10000],
        "processes_list": [["repr", "death", "compet"]],
        "mover_class": ["OU"],
        "save_temporal_data": [true],
        "save_positions": [false],
        "output_format": ["json"]
    },
    "filtering_parameter": "mover_class",
    "filters": {
        "OU": ["tau", "hr_stdev", "dispersal"],
        "BM": ["noise", "dispersal"],
        "SS": ["dispersal"]
    },
    "common_parameters": ["n_org", "..." ,"output_format"],
    "generate_seeds": true,
    "n_reps": 20,
    "id": "seed",
    "seed": 314159265,
    "filename": "exp_parameters.json"
}
```

The experiment files used to generate the data in the paper are available in the `experiment/` directory. A brief description of each file is provided below:

| File | Description | Figures using the data | # replicas | # simulations |
|-------|-------------|-----------------------|---------------|----------------|
| `...fig2.json` | Changing `hr_stdev` from $0.001$ to $1.0$ (19 levels) for three levels of `dispersal` ($0.001$, $0.01$, $0.1$) | Figure 2 and Supplementary Figure C.1 | 20 | 1260 |
| `...figs34.json` | Exploring the combinations of `hr_stdev` (from $0.001$ to $1.0$, 49 levels), `tau` (from $10^{-6}$ to $1.0$, 49 levels), `dispersal` ($0.001$, $0.01$, $0.1$), and `comp_kernel` ($0.001$, $0.01$, $0.1$) with a full factorial experiment design. | Figures 3 and 4, Supplementary Figures C.2 and C.5 | 20 | 432180 |
| `...gamma.json` | Changing `hr_stdev` from $0.001$ to $1.0$ (19 levels) for four levels of `dispersal`. Dispersal kernel was set to a gamma distribution with scale parameter $0.01$ and varying shape parameter ($1$, $2$, $4$, $8$)| Supplementary Figure C.3 | 50 | 3800 |

###  Generate Simulation Parameter Files:

Use the `run_experiments.py` script with the `-c` flag to create individual parameter files for each simulation run based on your experiment definition file.

```bash
python run_experiments.py -c -i experiments.json
```
This will generate parameter files (e.g., `exp_parameters.json`) containing the specific parameter combinations for each simulation.

### Run Simulations:

Execute the simulations using the generated parameter files. Run the `run_experiments.py` script with the `-r` flag for each parameter file.

```bash
python run_experiments.py -r -i exp_parameters.json
```
> [!WARNING]  
> Tests on an AMD Ryzen 9 3900X show that each simulation step takes 714 microseconds (single core). Thus, a simulation with $10^6$ steps would take approximately 12 minutes on a single core. The 1260 simulations required to generate the data for Figure 2 took 9h to complete when run in parallel using all 24 cores. The simulations required to generate the data for Figures 3 and 4 are estimated to take around 3570 CPU days if run sequentially. Therefore, it is highly recommended to run the simulations in parallel on a high-performance computing cluster.
> 
> Users should adjust the number of steps and repetitions in the `experiments.json` file according to their computational resources and desired accuracy.

### Aggregate Results:

Combine the results from all individual simulations into an aggregated dataset. Run the `run_experiments.py` script with the `-a` flag to aggregate the results, specifying the output directory where the individual results are stored with the `-o` flag.

```bash
python run_experiments.py -a -o path/to/your/output/
```

Replace `path/to/your/output/` with the actual path where the simulation results were saved.
The results can then be used to generate the figures.

## Acknowledgments

This work was financed, in part, by the São Paulo Research Foundation (FAPESP), Brasil - Process Number \#2024/18255-0; the National Council for Scientific and Technological Development, Brazil - CNPq: 140096/2021-3; the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001; This work was partially funded by the Center for Advanced Systems Understanding (CASUS), which is financed by Germany’s Federal Ministry of Education and Research (BMBF) and by the Saxon Ministry for Science, Culture and Tourism (SMWK) with tax funds on the basis of the budget approved by the Saxon State Parliament; the São Paulo Research Foundation (FAPESP, Brazil) through BIOTA Young Investigator Research Grant No. 2019/05523-8 (R.M-G); ICTP-SAIFR grant no. 2021/14335-0 (R.M. and R.M.-G); the Simons Foundation through grant no. 284558FY19 (R.M-G). The National Science Foundation (NSF, USA) grant DBI_al_1915347 supported the involvement of J.M.C. and W.F.F. This research was supported by resources supplied by the Center for Scientific Computing (NCC/GridUNESP) of the São Paulo State University (UNESP).

## License

This code is licensed under the MIT License

Copyright (c) 2025 Rafael Menezes

## Citation

If you use this code in your research, please cite the accompanying paper. The citation format is as follows:

```bibtex
@misc{menezes_range-resident_2025,
    title = {The range-resident logistic model: a new framework to formalize the population-dynamics consequences of range residency},
    shorttitle = {The range-resident logistic model},
    url = {https://www.biorxiv.org/content/10.1101/2025.02.09.637279v1},
    doi = {10.1101/2025.02.09.637279},
    language = {en},
    urldate = {2025-04-30},
    publisher = {bioRxiv},
    author = {Menezes, Rafael and Calabrese, Justin M. and Fagan, William F. and Prado, Paulo Inácio and Martinez-Garcia, Ricardo},
    year = {2025},
}
```
