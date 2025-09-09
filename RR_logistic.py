"""
@author: Rafael Menezes (github: r-menezes)
@date: 2025
@license: MIT License
"""

import os
from uuid import uuid4
from hashlib import sha256
import numpy as np
import math
from scipy.spatial import KDTree
import pandas as pd
# from statsmodels.tsa.stattools import adfuller
import gc
import copy

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from types import FunctionType#, UnionType
from typing import Any#, List, Dict, Tuple, Type
from itertools import repeat
import warnings
import json


# >>>> MOVEMENT CLASSES <<<<

class Mover:
    """A general class that other Mover functions inherit from."""

    def __init__(self, rng=np.random.default_rng(42)) -> None:
        self.rng = rng


    def move(self, pos, t):
        # returns the next position
        return self.__call__(pos, t)


    def update(self, pos, t):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )


    def relax_to_asymptotic_dist(self, initial_pos):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )


    def __call__(self, pos, t):
        # returns the next position
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

class SS_Mover(Mover):
    """An auxiliary class for sessile organisms."""

    def __init__(self, *args, **kwrgs) -> None:
        pass


    def __str__(self):
        """Returns the name of the mover"""
        return "SS"


    def update(self, habitat):
        pass


    def relax_to_asymptotic_dist(self, initial_pos):
        return initial_pos


    def __call__(self, pos, dt):
        return pos


class BM_Mover(Mover):
    """A class that implements Brownian motion."""

    def __init__(self, g, rng=np.random.default_rng(42), envsize=1.0, *args, **kwrgs) -> None:
        super().__init__(rng=rng)
        # the noise amplitude
        self.g = g
        self.envsize = envsize
        self.sqrt_g = np.sqrt(self.g)


    def __str__(self):
        """Returns the name of the mover"""
        return "BM"


    def update(self, habitat):
        pass


    def relax_to_asymptotic_dist(self, initial_pos):
        return initial_pos


    def __call__(self, pos, dt):
        new_pos = pos + self.rng.normal(size=pos.shape)*self.sqrt_g*math.sqrt(dt)
        return new_pos % self.envsize


class OU_Mover(Mover):
    """A class that implements Ornstein-Uhlenbeck motion."""

    def __init__(
        self,
        centers,
        g,
        tau,
        envsize=1.0,
        rng=np.random.default_rng(42),
    ) -> None:

        super().__init__(rng=rng)
        # centers of the OU process
        self.centers = centers
        # number of organisms
        self.n_orgs = len(centers)
        # the noise amplitude
        self.g = g
        # the average Home Range crossing time
        self.tau = tau
        # environment size
        self.envsize = envsize
        self.half_envsize = self.envsize / 2.0
        # derived parameters
        self.sigma2 = self.g * self.tau * 0.5
        self.invtau = 1.0 / self.tau
        self.invtau2 = - 2. * self.invtau


    def __str__(self):
        """Returns the name of the mover"""
        return "OU"


    def __call__(self, pos, dt):
        # find the closest image of the center
        # this is the direction the organism drifts towards
        center_image = self.find_closest_image(pos)

        # determine the mean and stdev of the distribution of the positions
        mean = (pos - center_image) * math.exp(-dt * self.invtau) + center_image
        stdev = math.sqrt(self.sigma2 * (1.0 - math.exp(dt * self.invtau2)))

        # sample new position and normalize it to the environment size
        new_pos = self.rng.normal(size=pos.shape)*stdev + mean
        new_pos = new_pos % self.envsize

        return new_pos


    def relax_to_asymptotic_dist(self, initial_pos, dt=1e3):
        # determine the mean and stdev of the distribution of the positions
        mean = initial_pos
        stdev = math.sqrt(self.sigma2 * (1.0 - math.exp(dt * self.invtau2)))

        # sample new position and normalize it to the environment size
        new_pos = self.rng.normal(size=initial_pos.shape)*stdev + mean
        new_pos = new_pos % self.envsize

        return new_pos


    def find_closest_image(self, pos):
        delta = self.centers - pos
        closest_disp = np.where(
            np.abs(delta) > self.half_envsize,
            delta - np.sign(delta) * self.envsize,
            delta,
        )
        # returns the index of the closest center
        return pos + closest_disp


    def update(self, habitat):
        self.centers = habitat.centers
        self.n_orgs = len(habitat.centers)


# >>>> AUXILIARY FUNCTIONS <<<<

class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types
    source: https://stackoverflow.com/a/65821617/13845224
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dict_to_arr(dic: dict, lis: list = None) -> np.ndarray:
    """Transform a dictionary into a numpy array."""
    if lis is None:
        return np.array([dic[k] for k in list(dic.keys())])
    else:
        return np.array([dic[k] for k in lis])

# >>>> HABITAT CLASS <<<<

@dataclass
class Habitat:
    # Initial populations
    n_org: int
    # Birth and death processes
    birth_rate: float
    death_rate: float
    # Competition process
    comp_kernel: float # (Gaussian) competition kernel standard deviation
    comp_max_dist: float # maximum distance for competition
    comp_rate: float  # competition rate between roaming organisms
    # Movement
    mover_class: Mover
    noise_amplitude: float # g
    tau: float # the average Home Range crossing time
    # Dispersal functions
    dispersal: FunctionType | float
    # Habitat Characteristics
    border: str  # "reflecting", "periodic", "open"
    env_size: float  # side of the square environment
    # Data variables
    data_interval: int | str = 'auto' # interval between data storage
    # 'auto' - store data every N steps, where N is the number of organisms
    # Processes
    processes_list: list[str] = field(default_factory=lambda: ["repr", "death", "compet"])
    continuous_processes_list: list[str] = field(default_factory=lambda: ["move"])
    # Time step
    no_rate_delta_t: float = 1e-2
    equilibrium_allowed: bool = False
    # HR centers initial distribution
    centers: np.ndarray | str = "random"
    # Random number generator
    rng: np.random.Generator | int = np.random.default_rng(16558947)
    # Unique id
    # hash - a hash of the parameters
    # uuid - a random uuid
    id: int | str = "hash"
    # Data
    extra_data: dict = field(default_factory=dict)
    save_temporal_data: bool = True
    save_positions: bool = False
    save_aggregates: bool = True
    save_output: bool = False
    output_format: str = "json"
    extra_data_on_exit: dict = field(default_factory=dict)
    temporal_averages: list[str] = field(default_factory=lambda: [])


    def __post_init__(self):
        """Initialization of habitat properties"""

        # save all the parameters in a dictionary
        self.params = self.__dict__.copy()

        # convert parameters to strings
        for k, v in self.params.items():
            if not isinstance(v, float) and not isinstance(v, int) and not isinstance(v, str):
                self.params[k] = str(v)

        # get info about Mover class
        aux = str(self.params["mover_class"])
        if 'BM' in aux:
            aux = 'BM'
        elif 'OU' in aux:
            aux = 'OU'
        elif 'SS' in aux:
            aux = 'SS'
        elif 'RD' in aux:
            aux = 'RD'
        self.params["mover_class"] = aux
        del aux

        # generate a unique id
        if self.id == "hash":
            # generate a hash of the parameters
            self.id = sha256(str(self.params).encode("utf-8")).hexdigest()
        elif self.id == "uuid":
            # generate a random uuid
            self.id = uuid4().hex
        else:
            # use the id passed by the user
            pass

        # Setup random number generator if a seed was passed
        if isinstance(self.rng, int) or isinstance(self.rng, np.integer):
            self.rng = np.random.default_rng(self.rng)

        # Environment properties
        self.t = 0.0
        self.steps = 0

        # Initalize established organisms HR center and positions
        if isinstance(self.centers, str):
            if self.centers == "random":
                self.centers = self.random_position(N=self.n_org)

            elif self.centers == "origin":
                self.centers = np.zeros((self.n_org, 2))

            else:
                raise ValueError(
                    "If the centers argument is a string, it must be either 'random' or 'origin'"
                )
        else:
            self.centers = np.array(self.centers)
            self.n_org = len(self.centers)

        self.pos = self.centers

        # Initialize the dispersal function
        if isinstance(self.dispersal, float):
            disp = copy.deepcopy(self.dispersal)
            self.dispersal = lambda pos, habitat: gaussian_dispersal(pos, habitat, disp)

        # Initialize the competition kernel
        if isinstance(self.comp_kernel, float):
            val = copy.deepcopy(self.comp_kernel)
            self.comp_kernel = lambda x: gaussian_kernel(x, val)

        # Initialize the movers
        self.mover = self.mover_class(
            centers=self.centers,
            g=self.noise_amplitude,
            tau=self.tau,
            envsize=self.env_size,
            rng=self.rng
        )

        # Relax each organism to its asymptotic position distribution (this is done to avoid the initial transient)
        self.pos = self.mover.relax_to_asymptotic_dist(initial_pos=self.pos)

        # interaction-related
        self.pos_tree = KDTree(
            self.pos,
            compact_nodes=False,
            balanced_tree=False,
            boxsize=self.env_size,
        )

        # pairs and rates of interaction
        self.interacting_pairs = None
        self.interacting_rates = None
        self.n_interacting_pairs = None

        # Gillespie variables
        self.total_rate = 0.0
        self.rates_dic = dict(
            zip(self.processes_list, repeat(0.0, len(self.processes_list)))
        )
        self.delta_t = 0.0

        # Stored data
        self.aggregates = {
            'N_org': 0,
            'repr': 0,
            'death': 0,
            'compet': 0
        }
        self.n_measurements = 0
        
        # If you have extra data, initialize it here
        if self.extra_data != {}:
            self.has_extra_data = True
            for key in self.extra_data.keys():
                self.aggregates[key] = 0
        else:
            self.has_extra_data = False

        # Storage properties
        self.n_events_dic = dict(
            zip(self.processes_list, repeat(0, len(self.processes_list)))
        )


    def run(self, max_steps, burn_in=0, max_abundance=np.inf, data_interval='auto', max_stored_data=np.inf, **kwargs):
        """Runs the simulation, calling gillespie_step for a number of gillespie steps given by `max_steps`.
        At every interval given by `data_interval`, the `store_data` method is called.
        Upon exit, the `on_exit` method is called and its output is returned.

        Args:
            max_steps (int): the number of gillespie steps that are going to be performed before the function returns with `timeout`.
            burn_in (int | float, optional): the burn-in period of the simulation. Default is 0.
            max_abundance (int, optional): the maximum abundance of organisms. Default is inf.
            data_interval (int | str, optional): the interval between data storage. Default is 'auto'.
            max_stored_data (int, optional): the maximum number of data points to store. Default is inf.

        Returns:
            pd.DataFrame: A pandas DataFrame with the output of the model.
        """
        from tqdm import tqdm

        if burn_in > max_steps:
            raise ValueError("The burn-in period is larger than the maximum number of simulation steps.")

        if data_interval == 'auto':
            data_interval = min(burn_in, 1)
            auto_interval = True
        else:
            auto_interval = False

        last_stored_data = 0
        n_stored_data = 0

        if isinstance(burn_in, float) and burn_in < 1:
            burn_in = int(burn_in * max_steps)

        self.steps = 0
        iterator = range(max_steps)
        if kwargs.get('tqdm', False):
            iterator = tqdm(range(max_steps))

        for _ in iterator:
            # perform a gillespie step
            self.gillespie_step()
            self.steps += 1

            # early exit conditions            
            if self.n_org < 1:
                return self.on_exit(cause="collapse")
            elif self.n_org > max_abundance:
                return self.on_exit(cause="max_abundance")

            # perform some_gillespie_steps as a burn-in period
            if self.steps < burn_in:
                continue

            # record data
            if (self.steps - last_stored_data) % data_interval == 0:

                n_stored_data += 1

                if auto_interval:
                    data_interval = max(self.n_org, 20)
                    last_stored_data = self.steps

                if self.save_temporal_data or self.save_aggregates:
                    self.store_data()

                # reset counts of number of events
                for proc in self.processes_list:
                    self.n_events_dic[proc] = 0

                if n_stored_data >= max_stored_data:
                    self.on_exit(cause="data")

        return self.on_exit(cause="timeout")


    def run_time(self, max_time, burn_in=0, max_abundance=np.inf, **kwargs):
        """Runs the simulation, calling gillespie_step for a number of gillespie steps given by `max_steps`.
        At every interval given by `data_interval`, the `store_data` method is called.
        Upon exit, the `on_exit` method is called and its output is returned.

        Args:
            max_time (float): the maximum time the system is going to be integrated by recursively performing gillespie steps before returning with `timeout`.
            burn_in (float, optional): the burn-in period of the simulation. Default is 0.
            max_abundance (int, optional): the maximum abundance of organisms. Default is inf.

        Returns:
            pd.DataFrame: A pandas DataFrame with the output of the model.
        """

        if burn_in > max_time:
            raise ValueError("The burn-in period is larger than the maximum time of the simulation.")

        if isinstance(burn_in, float) and burn_in < 1:
            burn_in = burn_in * max_time

        self.steps = 0
        while self.t < max_time:
            # perform a gillespie step
            self.gillespie_step()
            self.steps += 1

            # early exit conditions            
            if self.n_org < 1:
                return self.on_exit(cause="collapse")
            elif self.n_org > max_abundance:
                return self.on_exit(cause="max_abundance")

            # burn-in period (defined in terms of time)
            if self.t < burn_in:
                continue

            # record data
            if self.steps % self.data_interval == 0:
                if self.save_temporal_data or self.save_aggregates:
                    self.store_data()

                # reset counts of number of events
                for proc in self.processes_list:
                    self.n_events_dic[proc] = 0

        return self.on_exit(cause="timeout")


    def gillespie_step(self):
        """
        Performs a gillespie step:

        1) Calculates the current rates of all discrete processes.
        2) Sorts a time until the next discrete process from a exponential distribution.
        3) Select discrete process given their rates.
        4) Implements discrete stochastic process.
        5) Implements continuous processes [movement]
        6) updates simulation state (time).
        """

        # 1) calculate all possible interaction rates
        self.calculate_all_rates()

        # 2) calculate residence time
        if self.total_rate > 0:
            self.delta_t = - math.log(self.rng.random()) / self.total_rate
        else:
            self.delta_t = self.no_rate_delta_t

        # 3) choose process
        if len(self.processes_list) != 0:
            # choose process using a weighted random choice
            _probabilities = dict_to_arr(self.rates_dic, self.processes_list)
            _probabilities /= self.total_rate
            chosen_process = self.rng.choice(self.processes_list,
                p=_probabilities)
            
            # 4) Implement processes
            # implement birth
            if chosen_process == "repr":
                self.reproduction()

            # implement death
            elif chosen_process == "death":
                self.death()

            # implement death through competition
            elif chosen_process == "compet":
                self.competition()

            else:
                raise NotImplementedError(
                    f"No valid (Lagrangian) process was chosen by Gillespie algorithm. Revise 'gillespie_step()'.\n\
                    The chosen process was {chosen_process}."
                )
        
            # update counter
            self.n_events_dic[chosen_process] += 1

        # 5) Implements continuous processes

        for proc in self.continuous_processes_list:
            # implement movement
            if proc == "move":
                self.move(delta_t=self.delta_t)

            else:
                raise NotImplementedError(
                    f"No valid (continuous) process was chosen by Gillespie algorithm. Revise 'gillespie_step()'.\n\
                    The chosen process was {chosen_process}."
                )

        # 6) update time
        self.t += self.delta_t


    def calculate_all_rates(self, processes_list=None):
        # TODO: document this method

        if processes_list is None:
            processes_list = self.processes_list

        for proc in processes_list:            
            # calculate rates for each process and store them in a dictionary
            self.rates_dic[proc] = self.calc_rate(proc)
        
        # total rate
        self.total_rate = np.sum(
            dict_to_arr(dic=self.rates_dic, lis=processes_list)
        )


    def calc_rate(self, interaction_type):

        if interaction_type == "repr":
            return self.birth_rate * self.n_org

        elif interaction_type == "death":
            return self.death_rate * self.n_org

        elif interaction_type == "compet":

            # calculate the possible pairs of organisms that can interact
            # the pairs are stored in an array with two columns, each row is a pair with the index of the two organisms

            # create positions tree
            self.pos_tree = KDTree(
                self.pos,
                compact_nodes=False,
                balanced_tree=False,
                boxsize=self.env_size,
            )

            # Query KD tree for pairs of organisms
            self.interacting_pairs = self.pos_tree.query_pairs(
                r=self.comp_max_dist, eps=0.05, output_type="ndarray" # eps=0.05 => up to 5% error, but faster
            )
            self.n_interacting_pairs = len(self.interacting_pairs)

            # get an array with the squared distances between the pairs
            positions0 = self.pos[self.interacting_pairs[:, 0]]
            positions1 = self.pos[self.interacting_pairs[:, 1]]
            sqred_distances = self.sqred_distance_in_torus(
                x0=positions0, x1=positions1, size=self.env_size
            )

            # calculate the vector of interaction rates
            # In the simulation, this is implemented as the rate at which a pair of organisms interact and one of them dies.
            # Since in theory this is the rate at which EACH organism dies, the interaction rate in the simulation is multiplied by two.
            # The rate should be doubled to account for both organisms experiencing a symmetrical interaction.
            self.interacting_rates = (
                2 * self.comp_rate * self.comp_kernel(sqred_distances)
            )
            
            return np.sum(self.interacting_rates)

        else:
            raise NotImplementedError(
                f"Type of interacion not recognized passed to `calc_interaction_rates`. The interaction type passed as an argument was {interaction_type}."
            )


    def reproduction(self):
        """Method to implement reproduction."""
        # choose randomly which individual gave birth
        repr_id = self.rng.integers(self.n_org)
        # update birth
        self.add_organism(organism_id=repr_id)


    def death(self):
        """Method to implement death."""
        # choose randomly which individual died
        dead_id = self.rng.integers(self.n_org)
        # update death
        self.remove_organism(dead_id=dead_id)


    def competition(self):
        """Method to implement competition."""
        # choose randomly which pair interacted
        # probabilites = self.interacting_rates / self.rates_dic["compet"]
        competing_pair = self.rng.choice(self.n_interacting_pairs,
                p=self.interacting_rates/self.rates_dic["compet"])
        competing_pair = self.interacting_pairs[competing_pair]
        # choose randomly which individual of the pair dies
        r = self.rng.random()
        if r < 0.5:
            loser_id = competing_pair[0]
        else:
            loser_id = competing_pair[1]
        # update death
        self.remove_organism(dead_id=loser_id)


    @staticmethod
    def sqred_distance_in_torus(x0, x1, size):
        """returns the (squared) distance between two sets of points
        ref: https://stackoverflow.com/a/11109336/13845224

        Args:
            x0 (array): array with position of organisms
            x1 (array): array with position of organisms
            size (float / array): the size of the region with periodic boundary conditions

        Returns:
            array: distances between elements in vector x0 and x1
        """
        delta = np.abs(x0 - x1)
        delta = np.where(delta > 0.5 * size, delta - size, delta)
        return np.sum(delta * delta, axis=-1)


    def check_equilibrium(self):
        """Check for equilibrium in the timeseries data."""

        warnings.warn(
            "The equilibrium check is not implemented yet. The function will return False."
        )

        return False


    def add_organism(self, organism_id, type = None):
        # TODO: document this method
        # ignore organism 'type' for now

        # Disperse the center of the new organism
        mother_HR_center = self.centers[organism_id]
        daughter_HR_center = self.dispersal(mother_HR_center, self)

        # relax the position of the new organism to its asymptotic distribution
        daughter_pos = self.mover.relax_to_asymptotic_dist(initial_pos=daughter_HR_center)
        
        # store the new center and position
        self.centers = np.concatenate((self.centers, daughter_HR_center[np.newaxis]), axis=0)
        self.pos = np.concatenate((self.pos, daughter_pos[np.newaxis]), axis=0)

        # update the 'mover'
        self.mover.update(self)

        # increment number of individuals
        self.n_org += 1


    def remove_organism(self, dead_id, species = None):
        # ignore organism 'species' for now
        # if species == "E":

        # remove dead individual from centers and positions
        self.centers = np.delete(self.centers, dead_id, axis=0)
        self.pos = np.delete(self.pos, dead_id, axis=0)

        # decrement number of individuals and death counter
        self.n_org -= 1

        # update the 'mover'
        self.mover.update(self)


    def move(self, delta_t):
        # move all organisms
        self.pos = self.mover(self.pos, delta_t)
        self.pos = self.periodic_boundary_conditions(self.pos)


    def random_position(self, N=1):
        """Selects a random position for N organisms in the Habitat"""
        pos = self.rng.random(size=(N, 2))*self.env_size
        return pos


    def periodic_boundary_conditions(self, vector):
        return vector % self.env_size


    def on_exit(self, cause=None, **argv):
        # AVERAGE_N_ORGANISMS
        # each data point in self.data is assumed to be independent
        if cause == "collapse":
            ave_n_org = 0.0
        elif cause == "max_abundance":
            ave_n_org = self.n_org
        else:
            ave_n_org = self.aggregates['N_org'] / self.n_measurements if self.n_measurements > 0 else -1

        # RESULT DICT
        results = {
            "steps": self.steps,
            "final_time": self.t,
            "ave_N_org": ave_n_org,
            "final_N_org": self.n_org,
            "cause": cause,
            **argv,
        }

        # add extra data
        if self.extra_data_on_exit != {}:
            for key, fun in self.extra_data_on_exit.items():
                results[key] = fun(self)

        # add parameters
        results = results | self.params

        # AVERAGES
        # each data point in self.data is assumed to be independent
        if self.save_aggregates:
            averages = {}
            for key in self.aggregates.keys():
                if self.n_measurements > 0:
                    averages['ave_' + key] = self.aggregates[key] / self.n_measurements
                else:
                    averages['ave_' + key] = -1

            results = results | averages

        # FINAL VALUES
        # returns also the last stored value of each variable
        results['final_repr'] = self.n_events_dic['repr']
        results['final_death'] = self.n_events_dic['death']
        results['final_compet'] = self.n_events_dic['compet']

        if self.has_extra_data:
            for key, fun in self.extra_data.items():
                results['final_'+key] = fun(self)

        # create folders
        if self.save_output or self.save_temporal_data or self.save_positions:
            # create folders
            # output/ | output/results | output/centers | output/positions | output/temporalData
            self.create_folders()

        if self.save_output:
            # save results
            if self.output_format == "json":
                with open(f"output/results/results_{self.id}.json", "w") as f:
                    json.dump(results, f, cls=NumpyEncoder)
            else:
                res_df = pd.DataFrame(results, index=[0])

                if self.output_format == "parquet":
                    res_df.to_parquet(f"output/results/results_{self.id}.parquet")
                elif self.output_format == "csv":
                    res_df.to_csv(f"output/results/results_{self.id}.csv")
                # elif self.output_format == "json":
                #     res_df.to_json(f"output/results/results_{self.id}.json")
                elif self.output_format == "feather":
                    res_df.to_feather(f"output/results/results_{self.id}.feather")
                elif self.output_format == "hdf":
                    res_df.to_hdf(f"output/results/results_{self.id}.hdf")
                elif self.output_format == "pkl":
                    res_df.to_pickle(f"output/results/results_{self.id}.pkl")
                else:
                    raise ValueError(f"Output format {self.output_format} not recognized.")

        if self.save_positions:
            # save final positions as numpy array (npy)
            # save centers
            np.save(f"output/centers/centers_{self.id}.npy", self.centers)

            # save organism positions
            np.save(f"output/positions/positions_{self.id}.npy", self.pos)

        if self.save_temporal_data and hasattr(self, 'data') and self.data is not None:
            # save temporal data (parquet)
            self.data.reset_index(drop=True).to_parquet(
            f"output/temporalData/temporalData_{self.id}.parquet"
            )

        return results


    def create_folders(self, folders=None):
        """Creates the folders to store the output of the simulation"""
        import os

        # Default folders
        if folders is None:
            folders = ["output",
                        "output/results",
                        "output/centers",
                        "output/positions",
                        "output/temporalData"]

        # create folders
        for folder in folders:
            try:
                if not os.path.exists(folder):
                    os.makedirs(folder)
            except:
                pass


    def store_data(self):

        delta_aggregates = {
            'N_org': self.n_org,
            'repr': self.n_events_dic['repr'],
            'death': self.n_events_dic['death'],
            'compet': self.n_events_dic['compet'],
        }

        if self.has_extra_data:
            for key, fun in self.extra_data.items():
                delta_aggregates[key] = fun(self)

        # Update aggregates
        if self.save_aggregates:
            for key, value in delta_aggregates.items():
                if key not in self.aggregates:
                    self.aggregates[key] = 0
                self.aggregates[key] += value
        
        self.n_measurements += 1
        
        # Reset event counters
        for proc in self.processes_list:
            self.n_events_dic[proc] = 0

        # Add other essential information to the aggregates
        delta_aggregates['t'] = self.t
        delta_aggregates['steps'] = self.steps
        delta_aggregates['id'] = self.id

        # Store data in a pandas DataFrame
        if not hasattr(self, 'data'):
            self.data = pd.DataFrame()
        # Append the current aggregates to the data DataFrame
        self.data = pd.concat(
            [self.data, pd.DataFrame(delta_aggregates, index=[self.n_measurements-1])],
            ignore_index=True
        )

# >>>>> EXTRA DATA <<<<<<<

def hr_centers_crowding(habitat):
    """
    Calculates the interaction rate based on the home range centers of the organisms.

    Args:
        habitat (Habitat): the Habitat object

    Returns:
        float: the interaction rate, gamma, for the object
    """

    # calculate the possible pairs of organisms that can interact
    # the pairs are stored in an array with two columns, each row is a pair with the index of the two organisms

    if habitat.n_org < 2:
        return 0.

    # create positions tree
    tree = KDTree(
        habitat.centers,
        compact_nodes=False,
        balanced_tree=False,
        boxsize=habitat.env_size,
    )

    # define kernel properties
    try:
        mov_var = habitat.mover.sigma2
    except:
        print("Attempted to calculate HR centers interaction for wrong movement type? no sigma2 found in mover")
        return -1

    kernel_variance = habitat.params['comp_kernel']**2 + 2*mov_var
    kernel_stdev = np.sqrt(kernel_variance)
    max_dist = 3.0348542587702925 * kernel_stdev  # float # maximum distance for competition - 99% of the Rayleigh probability mass

    inter_kernel = lambda x: gaussian_kernel(x, kernel_stdev)

    # Query KD tree for pairs of organisms
    interacting_pairs = tree.query_pairs(
        r=max_dist, eps=0.01, output_type="ndarray" # eps=0.01 => up to 1% error, but faster
    )

    # get an array with the squared distances between the pairs
    positions0 = habitat.centers[interacting_pairs[:, 0]]
    positions1 = habitat.centers[interacting_pairs[:, 1]]
    all_sqred_distances = habitat.sqred_distance_in_torus(
        x0=positions0, x1=positions1, size=habitat.env_size
    )

    # filter distances
    sqred_distances = all_sqred_distances.flatten()
    sqred_distances = sqred_distances[sqred_distances < max_dist*max_dist]

    # calculate the vector of interaction rates
    # The rate should be doubled to account for both organisms experiencing a symmetrical interaction.
    interacting_rates = (
        2 * inter_kernel(sqred_distances)
    )

    # interaction rate
    # the interaction rate is the average interaction rate across all pairs
    rate = np.sum(interacting_rates)/(habitat.n_org*(habitat.n_org-1))
    # to convert from abundance rate to density, we must multiply by the area
    rate *= habitat.env_size**2

    # Effect of periodic boundary conditions (PBC) on the kernel normalization
    # because the competition between HRs can be very broad, the normalization of the kernel is off.
    # Essentially, integrating the kernel over the torus gives a value different from 1 if the kernel is very broad.
    # To correct for this, we renormalize, dividing by the result of integrating $\int_{-L/2}^{L/2} \int_{-L/2}^{L/2} K(x, y) dx dy = 2 \pi \sigma^2 (\operatorname{erf}(L/(2\sqrt{2}\sigma)))^2$
    # Because we use PBC, this factor is the same for all organisms/pairs of organisms.
    from math import erf
    _erf = erf(habitat.env_size / (2.828427125 * kernel_stdev))
    pbc_correction = 1./(_erf*_erf)

    return rate*pbc_correction


def org_pos_crowding(habitat):

    if habitat.n_org < 2:
        return 0.
    
    # calculate the possible pairs of organisms that can interact
    # the pairs are stored in an array with two columns, each row is a pair with the index of the two organisms

    # create positions tree
    tree = KDTree(
        habitat.pos,
        compact_nodes=False,
        balanced_tree=False,
        boxsize=habitat.env_size,
    )

    kernel_stdev = habitat.params['comp_kernel']
    max_dist = 3.0348542587702925 * kernel_stdev  # float # maximum distance for competition - 99% of the Rayleigh probability mass


    # Query KD tree for pairs of organisms
    interacting_pairs = tree.query_pairs(
        r=max_dist, eps=0.01, output_type="ndarray" # eps=0.01 => up to 1% error, but faster
    )

    # get an array with the squared distances between the pairs
    positions0 = habitat.pos[interacting_pairs[:, 0]]
    positions1 = habitat.pos[interacting_pairs[:, 1]]

    all_sqred_distances = habitat.sqred_distance_in_torus(
        x0=positions0, x1=positions1, size=habitat.env_size
    )

    # filter distances
    sqred_distances = all_sqred_distances.flatten()
    sqred_distances = sqred_distances[sqred_distances < max_dist*max_dist]

    # calculate the vector of interaction rates
    # In the simulation, this is implemented as the rate at which a pair of organisms interact and one of them dies.
    # Since in theory this is the rate at which EACH organism dies, the interaction rate in the simulation is multiplied by two.
    # The rate should be doubled to account for both organisms experiencing a symmetrical interaction.

    interacting_rates = (
        2 * gaussian_kernel(sqred_distances, kernel_stdev)
    )
    
    rate = np.sum(interacting_rates)/(habitat.n_org*(habitat.n_org-1))
    # to convert from abundance rate to density, we must multiply by the area
    rate *= habitat.env_size**2

    return rate

# >>>>>>> KERNELS <<<<<<<<

def gamma_dispersal(pos : np.ndarray,  habitat : Habitat, shape : float = 2.0, stdev : float = 0.1) -> np.ndarray:
    """This implements the gamma dispersal function for the roaming organisms.

    It assumes a dispersal distance that follows a gamma distribution.
    The dispersal direction is sampled uniformly from the unit circle.

    Args:
        pos (np.ndarray): the current position of the organism. A (1, 2) array.
        habitat (Habitat): the Habitat object.
        shape (float, optional): the shape parameter of the gamma distribution. Defaults to 2.0.
        stdev (float, optional): the standard deviation of the dispersal. Defaults to 0.1.

    Returns:
        np.ndarray: the position of the new HR center in the habitat
    """
    # sample new position and normalize it to the environment size
    n_new_positions = pos.shape[0] if pos.ndim > 1 else 1
    # sample distances from the gamma distribution
    dists = habitat.rng.gamma(shape=shape, scale=stdev, size=n_new_positions)
    angles = habitat.rng.uniform(low=0, high=2*np.pi, size=n_new_positions)
    # calculate new positions
    new_pos = pos + np.column_stack((dists * np.cos(angles), dists * np.sin(angles)))
    # reduce the number of dimensions if necessary
    if n_new_positions == 1:
        new_pos = new_pos[0]
    
    new_pos = habitat.periodic_boundary_conditions(new_pos)

    return new_pos

def gaussian_dispersal(pos : np.ndarray,  habitat : Habitat, stdev : float = 0.1) -> np.ndarray:
    """This implements the Gaussian dispersal function for the roaming organisms.

    Args:
        pos (np.ndarray): the current position of the organism. A (1, 2) array.
        stdev (float, optional): the standard deviation of the dispersal. Defaults to 0.1.

    Returns:
        np.ndarray: the position of the new HR center in the habitat
    """
    # sample new position and normalize it to the environment size
    new_pos = habitat.rng.normal(size=pos.shape)*stdev + pos
    new_pos = habitat.periodic_boundary_conditions(new_pos)

    return new_pos

def gaussian_kernel(dist_sqred, sigmaq):
    # the predation kernel is a bivariate Gaussian centered at zero
    # dist_sqred is a vector of squared distances between the organisms
    sigmaq_sq = sigmaq*sigmaq
    return np.exp(-dist_sqred/(2*sigmaq_sq))/(2*np.pi*sigmaq_sq)

# >>>>>>> MAIN FUNCTION <<<<<<<<

def main():
    assert (
        False
    ), "This is a module, not a script. Please import it and use the classes and functions defined in it."


if __name__ == "__main__":
    main()
