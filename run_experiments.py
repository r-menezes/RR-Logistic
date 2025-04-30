"""
@author: Rafael Menezes (github: r-menezes)
@date: 2025
@license: MIT License
"""

# # Running the Spatial Logistic Model

# ## Imports
# command line arguments
import argparse

# Import the classes created in SpatialDispersal
from RR_logistic import Habitat
from RR_logistic import BM_Mover, OU_Mover, SS_Mover


# Import numpy and pandas
import numpy as np
import pandas as pd
import json

# Parallel
from multiprocessing import Pool

# auxiliary Class
class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types
    source: https://stackoverflow.com/a/65821617/13845224
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Functions


def run_habitat(
    seed=16558947,
    steps=None,
    time=None,
    hr_stdev=None,
    tau=None,
    noise=None,
    mover_class=OU_Mover,
    default=False,
    max_abundance=np.inf,
    **kwargs
    ):
    from uuid import uuid4

    # Default parameters
    if default:
        tau = 1e-3
        hr_stdev = 1e-2
        steps = int(5e4)

    # select steps or time integration
    if (steps is None) and (time is None):
        raise ValueError("Either steps or time must be specified.")
    elif (steps is not None) and (time is not None):
        raise ValueError("Only one of steps or time can be specified.")

    time_integration = (time is not None)

    if time_integration and time == 'estimate':
        # Estimate the time needed for all processes to stabilize
        env_size = kwargs.get("env_size", 1.0)
        b = kwargs.get("birth_rate", 1.1)
        d = kwargs.get("death_rate", 0.1)
        dispersal = kwargs.get("dispersal", 0.1)
        tdisp = .5*env_size/(np.sqrt(b*(b-d))*dispersal)

        # time for the population to reach K in the Mean Field approximation
        # exp(-3) ~ 0.05, so the system roughly relaxed to the equilibrium already
        tdem = 3./(b-d)

        time = max(tdisp, tdem)

    #  Movement parameters
    if tau is None:
        tau = kwargs.get("tau", None)

    if mover_class == "OU" or mover_class is OU_Mover:
        mover_class = OU_Mover
        if (tau is None) + (noise is None) + (hr_stdev is None) != 1:
            print(f"tau: {tau}, noise: {noise}, hr_stdev: {hr_stdev}")
            raise ValueError(
                "Exactly two of the parameters tau, noise, hr_stdev must be specified for OU movement"
            )

        # Relationship between tau, noise and hr_stdev
        # hr_stdev^2 = noise*tau/2
        if tau is None:
            tau = 2.0 * hr_stdev**2 / noise
        elif noise is None:
            noise = 2.0 * hr_stdev**2 / tau
        elif hr_stdev is None:
            hr_stdev = np.sqrt(noise * tau / 2.0)

    elif mover_class == "BM" or mover_class is BM_Mover:
        mover_class = BM_Mover
        if noise is None:
            raise ValueError("The parameter noise must be specified for BM movement")

    elif mover_class == "SS" or mover_class is SS_Mover:
        mover_class = SS_Mover

    elif mover_class == "RD" or mover_class is RD_Mover:
        mover_class = RD_Mover

    # Birth and death processes
    b = kwargs.get(
        "birth_rate", 1.1
        ) # birth_rate: float
    d = kwargs.get(
        "death_rate", 0.1
        ) # death_rate: float

    # # Habitat Characteristics
    border = kwargs.get(
        "border", "periodic"
    )  #: str  # "reflecting", "periodic", "open"
    env_size = kwargs.get("env_size", 1.0)  #: float  # side of the square environment

    # Competition process
    comp_kernel = kwargs.get(
        "comp_kernel", 3e-2
    )  # FunctionType | float # competition kernel
    comp_max_dist = 2.447746830680816 * comp_kernel  # float # maximum distance for competition - 95% of the Rayleigh probability mass
    gamma = kwargs.get(
        "comp_rate", 0.02
    )  # float  # competition rate between roaming organisms
    gamma = gamma / (env_size*env_size)

    # # Movement
    mover_class = mover_class  #: Mover
    noise_amplitude = noise  # float # g
    tau = tau  #: float # the average Home Range crossing time

    # # Dispersal functions
    dispersal = kwargs.get("dispersal", 0.1)  #: FunctionType | float

    # Initial populations
    n_org = kwargs.get("n_org", int(round((b - d) / gamma)))  #: int
    centers = kwargs.get("centers", "random")  #: np.ndarray | str = "random"

    # # Data variables
    data_interval = kwargs.get("data_interval", 50)  #: int

    # # Processes
    processes_list = kwargs.get(
        "processes_list", ["repr", "death", "compet"]
    )  #: list[str] = field(default_factory=lambda: ["repr", "death", "compet"])

    continuous_processes_list = kwargs.get(
        "continuous_processes_list", ["move"]
    )  #: list[str] = field(default_factory=lambda: ["move"])

    # # Data
    rng = int(seed)  #: np.random.Generator | int = np.random.default_rng(16558947)
    no_rate_delta_t = kwargs.get("no_rate_delta_t", 1e-2)  #: float = 1e-2
    extra_data = kwargs.get("extra_data", {})  #: dict = field(default_factory=dict)
    temporal_averages = kwargs.get("temporal_averages", None)
    return_temporal_data = kwargs.get("return_temporal_data", False)  #: bool = False
    return_positions = kwargs.get("return_positions", False)  #: bool = False
    save_temporal_data = kwargs.get("save_temporal_data", True)  #: bool = False
    save_positions = kwargs.get("save_positions", True)  #: bool = False
    save_output = kwargs.get("save_output", True)  #: bool = False
    output_format = kwargs.get("output_format", "parquet")  #: str = "json"
    extra_data_on_exit = kwargs.get(
        "extra_data_on_exit", {}
    )  #: dict = field(default_factory=dict)

    burn_in = kwargs.get("burn_in", 0.5 ) #: int | float = 0.5 # 50% burn in
    # burn in can be specified as a fraction of the total time or steps
    if isinstance(burn_in, float) and burn_in < 1:
        if time_integration:
            burn_in = burn_in * time
        else:
            burn_in = int(burn_in * steps)

    # # ID
    _id = kwargs.get("id", "uuid")  #: int | str = "hash"

    from RR_logistic import hr_centers_crowding, org_pos_crowding

    extra_data = {"hr_crowd": hr_centers_crowding, "pos_crowd": org_pos_crowding}
    temporal_averages = ["hr_crowd", "pos_crowd"]

    habitat = Habitat(
        n_org=n_org,
        birth_rate=b,
        death_rate=d,
        comp_kernel=comp_kernel,
        comp_max_dist=comp_max_dist,
        comp_rate=gamma,
        mover_class=mover_class,
        noise_amplitude=noise_amplitude,
        tau=tau,
        dispersal=dispersal,
        border=border,
        env_size=env_size,
        data_interval=data_interval,
        processes_list=processes_list,
        continuous_processes_list=continuous_processes_list,
        no_rate_delta_t=no_rate_delta_t,
        centers=centers,
        rng=rng,
        extra_data=extra_data,
        return_temporal_data=return_temporal_data,
        return_positions=return_positions,
        save_temporal_data=save_temporal_data,
        save_positions=save_positions,
        save_output=save_output,
        output_format=output_format,
        extra_data_on_exit=extra_data_on_exit,
        temporal_averages=temporal_averages,
        id=_id,
        equilibrium_allowed=False
    )

    try:
        if time_integration:
            out = habitat.run_time(time, burn_in=burn_in)
        else:
            out = habitat.run(steps,
                burn_in=burn_in,
                max_abundance=max_abundance,
                data_interval=data_interval)
    except Exception as e:

        print(">>> Error in run_habitat: <<<")

        # print the traceback
        import traceback
        traceback.print_exc()

        # Print the error message
        print(f"Error: {e}")

        # print the parameters of the experiment, in a pretty way
        print("\nParameters:")
        for k, v in kwargs.items():
            print(f"{k}: {v}")

        print("\n<<<>>>")
        out = {}

    return out


def experiment_planner(
    parameter_levels: dict[str:list],
    common_parameters: list = None,
    filtering_parameter: str = None,
    filters: dict[str:list] = None,
    generate_seeds: bool = False,
    n_reps: int = 10,
    seed: int = 16558947,
    id: str = None,
    filename: str = None,
    randomize: bool = True,
) -> list[dict]:
    """
    The experiment_planner function is used to generate full factorial experiments with the ability to filter parameter levels based on a specified filtering parameter. It returns a list of dictionaries, where each dictionary represents a single experiment with parameter-value pairs.

    Parameters

    parameter_levels (dict[str: list]): A dictionary specifying the parameter names as keys and their corresponding levels as lists. Each key-value pair represents a parameter and its possible levels.
    filtering_parameter (str, optional): The parameter name used for filtering. If specified, only the parameter levels compatible with the filters will be included in the experiments. Default is None.
    filters (dict[str: list], optional): A dictionary specifying the filtering levels for the filtering parameter. The keys represent the filtering levels, and the values are lists of parameter names that are compatible with each level. This parameter is required if filtering_parameter is specified. Default is None.
    generate_seeds (bool, optional): A flag indicating whether to assign random seeds to the experiments. If True, each experiment will have a 'seed' key with a randomly generated seed value. Default is False.
    n_reps (int, optional): The number of repetitions for each experiment, only used if `generate_seeds = True`. Default is 10.
    id (bool, optional): A flag indicating whether to assign a unique id to each experiment. If True, each experiment will have an 'id' key with a random uuid4 value. Default is False.
    seed (int, optional): The seed used to generate the random seeds. Only used if `generate_seeds = True`. Default is 16558947.

    Returns

    experiments (list[dict]): A list of dictionaries representing the generated experiments. Each dictionary contains parameter-value pairs.

    Example Usage

    # Example 1: Generate full factorial experiments without filtering
    parameter_levels = {
        'param1': [1, 2, 3],
        'param2': ['A', 'B']
    }
    experiments = experiment_planner(parameter_levels)
    print(experiments)
    # Output:
    # [{'param1': 1, 'param2': 'A'}, {'param1': 1, 'param2': 'B'}, {'param1': 2, 'param2': 'A'},
    #  {'param1': 2, 'param2': 'B'}, {'param1': 3, 'param2': 'A'}, {'param1': 3, 'param2': 'B'}]

    # Example 2: Generate filtered experiments based on a filtering parameter
    parameter_levels = {
        'param1': [1, 2, 3],
        'param2': ['A', 'B', 'C']
    }
    filters = {
        'filter1': ['param1', 'param2'],
        'filter2': ['param2']
    }
    experiments = experiment_planner(parameter_levels, filtering_parameter='filter1', filters=filters)
    print(experiments)
    # Output:
    # [{'param1': 1, 'param2': 'A'}, {'param1': 1, 'param2': 'B'}, {'param1': 1, 'param2': 'C'},
    #  {'param1': 2, 'param2': 'A'}, {'param1': 2, 'param2': 'B'}, {'param1': 2, 'param2': 'C'},
    #  {'param1': 3, 'param2': 'A'}, {'param1': 3, 'param2': 'B'}, {'param1': 3, 'param2': 'C'}]

    # Example 3: Generate filtered experiments with assigned random seeds
    parameter_levels = {
        'param1': [1, 2],
        'param2': ['A', 'B']
    }
    experiments = experiment_planner(parameter_levels, generate_seeds=True, n_reps=1)
    print(experiments)
    # Output:
    # [{'param1': 1, 'param2': 'A', 'seed': 1385630157}, {'param1': 2, 'param2': 'B', 'seed': 1604595086}]
    """
    import itertools
    import numpy as np
    from uuid import uuid4

    if filtering_parameter is None:
        # *dict.items() unpacks the dictionary into a list of (key, value) tuples
        # zip(*list) unpacks the list of (key, value) tuples into (key_list, value_list)
        pars, values = zip(*parameter_levels.items())

        # create a list of dictionaries with all combinations of parameters
        experiments = [dict(zip(pars, v)) for v in itertools.product(*values)]

    elif filters is None:
        raise ValueError(
            "If filtering_parameter is specified, filters must be specified too."
        )

    else:
        all_experiments = []

        # get the levels of the filtering parameter
        filtering_levels = parameter_levels.get(filtering_parameter).copy()

        for level in filtering_levels:
            # filter the parameters so that only the parameters accepted by the filtering level are kept
            # i.e. if filtering_parameter = 'movement', filtering_level = 'OU', then only the parameters
            # accepted by the OU_Mover class are kept
            filtered_parameter_levels = {
                k: v for k, v in parameter_levels.items() if (k in filters[level] or k in common_parameters)
            }

            # create a list of filtered parameters and values
            pars, values = zip(*filtered_parameter_levels.items())

            # create a list of dictionaries with all combinations of parameters
            experiments = [dict(zip(pars, v)) for v in itertools.product(*values)]

            # add the filtering parameter to the dictionary
            for exp in experiments:
                exp[filtering_parameter] = level

            all_experiments.extend(experiments)

        experiments = all_experiments

    if generate_seeds:
        # spawn all seeds if requested
        ss = np.random.SeedSequence(seed)
        seeds = ss.generate_state(len(experiments) * n_reps)

        # repeat experiments n_reps times and then chain them into a single list
        repeated_experiments = list(
            itertools.chain(*itertools.repeat(experiments, n_reps))
        )

        for idx, exp in enumerate(repeated_experiments):
            repeated_experiments[idx] = exp | {"seed": seeds[idx]}

        experiments = repeated_experiments

    if id is not None:
        if id == "uuid":
            # BUG: uuid4() is not reproducible
            # BUG: uuid is too large to be stored as an int
            # add a unique random id to each experiment
            for idx, exp in enumerate(experiments):
                experiments[idx] = exp | {"id": str(uuid4().hex)}

        if id == "hash":
            # create an id by hashing the string representation of the experiment
            # if hash is negative, we add a zero to the left
            for idx, exp in enumerate(experiments):
                _hash = hash(str(exp))
                _hash = str(_hash) if _hash > 0 else "0" + str(abs(_hash))
                experiments[idx] = exp | {"id": _hash}

        if id == "index":
            # use the index of the experiment in the list as an id
            for idx, exp in enumerate(experiments):
                experiments[idx] = exp | {"id": str(idx)}

        if id == "seed":
            # use the seed as an id
            for idx, exp in enumerate(experiments):
                experiments[idx] = exp | {"id": str(exp["seed"])}

    if randomize:
        # randomize the order of the experiments
        np.random.shuffle(experiments)

    if filename is not None:
        with open(filename, "w") as f:
            json.dump(experiments, f, cls=NumpyEncoder)

    print(f"{len(experiments)} generated.")

    return experiments


# function to read the input for the experiment planner from a json file
def read_experiment_planner_input(filename: str) -> dict:
    """
    The read_experiment_planner_input function is used to read the input for the experiment planner from a json file. It returns a tuple with four elements: the first element is a dictionary containing the parameter levels, the second element is the filtering parameter, the third element is a dictionary containing the filters, and the fourth element is a boolean indicating whether to assign random seeds to the experiments.

    Parameters

    filename (str): The name of the json file containing the experiment planner input.

    Returns

    (parameter_levels, filtering_parameter, filters, seeds) (tuple[dict, str, dict, bool]): A tuple containing the parameter levels, filtering parameter, filters, and seeds.

    Example Usage

    # Example 1: Read input from a json file

    # The contents of experiment_planner_input.json are:
    # {
    #     "parameter_levels": {
    #         "tau": [0.1, 0.5, 1.0, 2.0, 5.0],
    #         "noise": [0.01, 0.02, 0.04, 0.08, 0.16],
    #         "hr_stdev": [0.01, 0.02, 0.04, 0.08, 0.16],
    #         "n_org": [100, 1000, 10000],
    #         "movement": ["OU", "BM", "SS"]
    #     },
    #     "filtering_parameter": "movement",
    #     "filters": {
    #         "OU": ["tau", "noise", "n_org"],
    #         "BM": ["noise", "n_org"],
    #         "SS": ["n_org"]
    #     "seeds": true
    # }
    """
    import json
    import os

    if not os.path.isfile(filename):
        filename = os.path.join(os.getcwd(), filename)

    if not os.path.isfile(filename):
        raise ValueError("The filename argument must be a json file.")

    with open(filename) as f:
        data = json.load(f)

    # validate the input
    if not isinstance(data, dict):
        raise ValueError("The input must be a dictionary.")

    if "parameter_levels" not in data:
        raise ValueError("The input must contain the parameter_levels key.")

    if not isinstance(data["parameter_levels"], dict):
        raise ValueError("The parameter_levels key must contain a dictionary.")

    if (
        "filtering_parameter" in data
        and data["filtering_parameter"] not in data["parameter_levels"]
    ):
        raise ValueError("The filtering_parameter must be one of the parameter levels.")

    if "filtering_parameter" in data and "filters" not in data:
        raise ValueError(
            "If filtering_parameter is specified, filters must be specified too."
        )

    if "filters" in data and not isinstance(data["filters"], dict):
        raise ValueError("The filters key must contain a dictionary.")

    if "generate_seeds" in data and not isinstance(data["generate_seeds"], bool):
        raise ValueError("The generate_seeds key must be True or False.")

    if "generate_seeds" not in data:
        data["generate_seeds"] = False

    return data


# function to run all experiments in parallel using the multiprocessing library
def run_full_factorial_p(experiments):
    from multiprocessing import Pool
    from os import cpu_count

    _run_exp = lambda exp: run_habitat(**exp)

    from p_tqdm import p_map

    # run the model for all experiments
    num_cores = cpu_count()

    print(f"Using {num_cores} cores to run the experiments.")

    results = p_map(_run_exp,
                experiments,
                desc='\n')

    return results

def aggregate_json_files(folder):
    """Aggregate all json files in a folder into a single json file"""
    import json
    import os

    if not os.path.isdir(folder):
        folder = os.path.join(os.getcwd(), folder)

    if not os.path.isdir(folder):
        raise ValueError("The folder argument must be a directory.")

    # get all json files in the folder
    json_files = [
        pos_json for pos_json in os.listdir(folder) if pos_json.endswith(".json")
    ]

    # create a list of dictionaries
    json_list = []
    for json_file in json_files:
        with open(os.path.join(folder, json_file)) as f:
            json_list.append(json.load(f))

    # write the list of dictionaries to a json file
    with open(os.path.join(folder, "aggregated.json"), "w") as f:
        json.dump(json_list, f)


def categorize_columns(df):
    """Categorize non-numeric columns of a pandas dataframe"""
    import pandas as pd

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")

    return df

def file_generator(folder):
    """A generator that yields all files in a folder"""
    import os

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            yield file_path

def save_dataframe(df, folder="output", filename="aggregated.parquet"):
    """
    Save a pandas dataframe to a file.
    """
    import os
    import pandas as pd

    file_extension = os.path.splitext(filename)[1]
    if file_extension == ".parquet":
        df.to_parquet(os.path.join(folder, filename))
    elif file_extension == ".csv":
        df.to_csv(os.path.join(folder, filename))
    elif file_extension == ".json":
        df.to_json(os.path.join(folder, filename))
    elif file_extension == ".feather":
        df.to_feather(os.path.join(folder, filename))
    elif file_extension == ".hdf":
        df.to_hdf(os.path.join(folder, filename))
    elif file_extension == ".pkl":
        df.to_pickle(os.path.join(folder, filename))
    else:
        # warn user if the file extension is not one of the supported types
        print(f"Warning: {file_extension} is not a supported file type.")
        print("Using parquet as default.")
        name = os.path.splitext(filename)[0]
        filename = name + ".parquet"
        df.to_parquet(os.path.join(folder, filename))

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

def aggregate_with_pandas(input_obj, save=False, delete_old=False, folder="output", filename="aggregated.parquet"):
    """
    Aggregate a list of dictionaries or multiple files into a pandas dataframe.

    Parameters:
        input_obj (list or str): The input object to aggregate. It can be a list of dictionaries or a folder path.
        save (bool, optional): Whether to save the resulting dataframe to a parquet file. Defaults to False.
        delete_old (bool, optional): Whether to delete the old files after aggregating them. Defaults to False.
        folder (str, optional): The folder path to save the parquet file. Defaults to "output".
        filename (str, optional): The name of the parquet file. Defaults to "aggregated.parquet".

    Returns:
        pandas.DataFrame: The aggregated dataframe.

    Raises:
        ValueError: If the folder argument is not a directory or if the input_obj argument is not a list of dictionaries or a folder.
    """
    import pandas as pd
    from tqdm import tqdm
    from dask import dataframe as dd
    from dask.diagnostics import ProgressBar
    import os

    print("Aggregating data | Registering progress bar...")
    # progress bar for dask
    pbar = ProgressBar()
    pbar.register()

    if not os.path.isdir(folder):
        folder = os.path.join(os.getcwd(), folder)

    if not os.path.isdir(folder):
        raise ValueError("The folder argument must be a directory.")

    if isinstance(input_obj, list):
        df = pd.DataFrame(input_obj)

    elif isinstance(input_obj, str):
        if not os.path.isdir(input_obj):
            input_obj = os.path.join(os.getcwd(), input_obj)

        if not os.path.isdir(input_obj):
            raise ValueError("The input_obj argument must be a list of dictionaries or a folder.")

    else:
        raise ValueError("The input_obj argument must be a list of dictionaries or a folder.")

    # assume is a folder
    print("Aggregating data | read_parquet...")
    df = dd.read_parquet(input_obj)
    print("Aggregating data | compute...")
    df = df.compute()

    print("Aggregating data | post-processing...")
    # Categorize the columns
    df = df.fillna(value=np.nan)
    df = df.replace('None', np.nan)
    df = categorize_columns(df)

    # Write dataframe to file
    if save:
        print("Aggregating data | saving dataframe...")
        save_dataframe(df, folder=folder, filename=filename)

    # unregister progress bar
    pbar.unregister()

    return df

def main():
    from time import ctime
    import sys
    import json

    parser = argparse.ArgumentParser(
                    prog='RunHomeRangeLogisticExperiments',
                    description='This is a series of helper functions to create and run experiments for the Range-Resident Logistic Model.',
                    epilog='2025 CC-BY Rafael Menezes (github: r-menezes)')
    # input/output files
    parser.add_argument('-i', '--input', type=str, help='Path of JSON file containing the experiments to be run if in run mode, the JSON with experiments specification if in create mode, the JSON with experiments to prune if in prune mode', default='experiments.json')
    parser.add_argument('-o', '--output', type=str, help='Output folder path', default='output/')

    # Specific Arguments
    parser.add_argument('--cleaned', type=str, help='Cleaned experiments file path', default='exp_parameters_cleaned.json')
    parser.add_argument('--runs_per_file', type=int, help='Number of runs per file', default=500)

    # actions
    parser.add_argument('-c', '--create', action='store_true', help='Create experiments')
    parser.add_argument('-r', '--run', action='store_true', help='Run experiments')
    parser.add_argument('-a', '--aggregate', action='store_true', help='Aggregate results')
    parser.add_argument('-s', '--split', action='store_true', help='Split the input file into multiple files for parallel processing')

    # testing
    parser.add_argument('-t', '--test', action='store_true', help='Test the code by running it with default parameters')

    # parse arguments
    args = parser.parse_args()

    # Can only run one mode at a time
    count_true = args.create + args.run + args.aggregate + args.split + args.test
    if count_true > 1:
        print("Error: Only one mode can be selected. Use -c, -r or -p.")
        parser.print_help()
        sys.exit()

    elif count_true == 0:
        # If no mode is selected, default to test
        args.create = False
        args.run = False
        args.aggregate = False
        args.split = False
        args.test = True
        print("No mode selected. Defaulting to test mode.")
        parser.print_help()

    # Create experiments
    if args.create:
        now = ctime()
        print(f"{now} || Creating experiments specified in {args.input}...")
        # read the input from the json file
        data = read_experiment_planner_input(args.input)

        # create the experiments
        experiments = experiment_planner(**data)

        # set the input to the output file
        args.input = data['filename']

    if args.run:
        now = ctime()
        print(f"{now} || Running experiments specified in file {args.input}...")
        # load experiments
        with open(args.input) as f:
            experiments = json.load(f)

        # run the model for all experiments
        results = run_full_factorial_p(experiments)

    if args.aggregate:
        # aggregate results
        now = ctime()
        print(f"{now} || Aggregating results...")

        input_folder = args.output + "/results"
        # aggregate results
        df = aggregate_with_pandas(input_folder, folder=args.output, save=True, delete_old=False)
        # print summary statistics
        print(df.describe())

    if args.split:
        import math
        from random import shuffle
        import json
        import os

        now = ctime()
        print(f"{now} || Splitting {args.input} into multiple files...")

        # Load exp_parameters_filtered
        with open(args.input, 'r') as f:
            exp_params = json.load(f)

        print(f"Total number of experiments: {len(exp_params)}")

        # Calculate size of each chunk
        chunk_size = args.runs_per_file

        # Number of files to split into
        N = math.ceil(len(exp_params) / chunk_size)
        print(f"Number of files to split into: {N}")

        # Shuffle experiments
        # this ensures that the amount of work will be evenly split along the jobs
        shuffle(exp_params)

        # Divide exp_parameters_filtered into chunks
        # note: if the final index in the range is greater than the length of the list, it will return the list until the end
        chunks = [exp_params[i:i + chunk_size] for i in range(0, len(exp_params), chunk_size)]

        # Check if folder exists
        if not os.path.isdir(args.output):
            os.mkdir(args.output)

        # Save each chunk into a separate file
        for i, chunk in enumerate(chunks):
            with open(args.output + str(args.input).replace(".json", f"_{i}.json"), 'w') as f:
                json.dump(chunk, f)

    if args.test:
        now = ctime()
        print(f"{now} || Running test with default parameters...")

        # Define default test parameters
        test_params = {
            'seed': 16558947,
            'steps': int(5e4),
            'data_interval': 500,
            'burn_in': int(3e4),
            'mover_class': OU_Mover,
            'default': True
        }

        # Run the experiment
        result = run_habitat(**test_params)

        # Print the output
        print("Test result:")
        print(json.dumps(result, cls=NumpyEncoder, indent=4))
        # Save the output
        with open(args.output + "test_result.json", 'w') as f:
            json.dump(result, f, cls=NumpyEncoder, indent=4)
        print(f"Test result saved to {args.output}test_result.json")


if __name__ == "__main__":
    main()
