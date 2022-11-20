#!/usr/bin/python
"""

In this example we evolve running soft robots in a terrestrial environment using a standard version of the physics
engine (_voxcad). After running this program for some time, you can start having a look at some of the evolved
morphologies and behaviors by opening up some of the generated .vxa (e.g. those in
evosoro/evosoro/basic_data/bestSoFar/fitOnly) with ./evosoro/evosoro/_voxcad/release/VoxCad
(then selecting the desired .vxa file from "File -> Import -> Simulation")

The phenotype is here based on a discrete, predefined palette of materials, which are visualized with different colors
when robots are simulated in the GUI.

Materials are identified through a material ID:
0: empty voxel, 1: passiveSoft (light blue), 2: passiveHard (blue), 3: active+ (red), 4:active- (green)

Active+ and Active- voxels are in counter-phase.

"""
import random
import numpy as np
import subprocess as sub
import os
import sys
import argparse

# Appending repo's root dir in the python path to enable subsequent imports
sys.path.append(os.getcwd() + "/../..")

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.model import CA
from evosoro.cellbot import CellBotGenotype, CellBotPhenotype, CellBotPopulation
from evosoro.tools.algorithms import ParetoOptimization
from evosoro.tools.checkpointing import continue_from_checkpoint
from evosoro.tools.mutation import create_new_children_through_mutation_cell


VOXELYZE_VERSION = '_voxcad_cluster'
# sub.call("rm ./voxelyze", shell=True)
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)  # Making sure to have the most up-to-date version of the Voxelyze physics engine
# sub.call("chmod 755 ./voxelyze", shell=True)
# sub.call("cp ../" + VOXELYZE_VERISON + "/qhull .", shell=True)  # Auxiliary qhull executable, used in some experiments to compute the convex hull of the robot
# sub.call("chmod 755 ./qhull", shell=True)  # Execution right for qhull


NUM_RANDOM_INDS = 1  # Number of random individuals to insert each generation
MAX_GENS = 1000 # Number of generations
POPSIZE = 30  # Population size (number of individuals in the population)
IND_SIZE = (7, 7, 7)  # Bounding box dimensions (x,y,z). e.g. IND_SIZE = (6, 6, 6) -> workspace is a cube of 6x6x6 voxels
SIM_TIME = 5  # (seconds), including INIT_TIME!
INIT_TIME = 0.5
DT_FRAC = 0.9  # Fraction of the optimal integration step. The lower, the more stable (and slower) the simulation.

TIME_TO_TRY_AGAIN = 30  # (seconds) wait this long before assuming simulation crashed and resending
MAX_EVAL_TIME = 60  # (seconds) wait this long before giving up on evaluating this individual
SAVE_LINEAGES = False
MAX_TIME = 36  # (hours) how long to wait before autosuspending
EXTRA_GENS = 0  # extra gens to run when continuing from checkpoint

RUN_DIR = "stable7x7x7"  # Subdirectory where results are going to be generated
RUN_NAME = "Stable7x7x7"
CHECKPOINT_EVERY = 100  # How often to save an snapshot of the execution state to later resume the algorithm
SAVE_POPULATION_EVERY = 100  # How often (every x generations) we save a snapshot of the evolving population
PLOT_FITNESS_EVERY = 100 # How often to plot the max and mean fitness

EVAL_STAGE = 10 # How many growth stages of the cellular automata before the phenotype is evaluated
STABLE_STAGES = [11,12,13,14,15] # Which stages need to be stable



# And, finally, our main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='N',
      help='seed used for generating random numbers')
    args = parser.parse_args()
    random.seed(args.seed)  # Initializing the random number generator for reproducibility
    np.random.seed(args.seed)
    RUN_DIR = 'simulation_data/' + RUN_DIR + '_' + str(args.seed)
    
    
    # Defining a custom genotype, inheriting from base class Genotype
    class MyGenotype(CellBotGenotype):
        def __init__(self, orig_size_xyz=IND_SIZE):
            # We instantiate a new genotype for each individual which must have the following properties
            model = CA(orig_size_xyz)
            CellBotGenotype.__init__(self, model)

    # Define a custom phenotype, inheriting from the Phenotype class
    class MyPhenotype(CellBotPhenotype):
        
        def __init__(self, genotype, stable_stages=STABLE_STAGES):
            self.stable_stages = stable_stages
            CellBotPhenotype.__init__(self, genotype, eval_stage=EVAL_STAGE)
            
        def initialise(self):
            a = np.zeros(shape=self.genotype.model.robot_shape)
            np.put(a,a.size//2,1)
            self.state_history = [a]
            self.alpha_history = [a]
            self.grow(max([self.eval_stage] + self.stable_stages)-1)
            self.size = np.count_nonzero(self.eval_state)

            
        def _get_instability(self):
            """Calculate how many differences there are between the eval stage and each stable stage"""
            target = self.state_history[self.eval_stage-1]
            changes = 0
            for ss in self.stable_stages:
                changes = np.sum(target != self.state_history[ss-1])
            return changes
            
        instability = property(fget=_get_instability)        
            
            

    # Setting up the simulation object
    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

    # Setting up the environment object
    my_env = Env(sticky_floor=0, time_between_traces=0)

    # Now specifying the objectives for the optimization.
    # Creating an objectives dictionary
    my_objective_dict = ObjectiveDict()

    # Adding an objective named "fitness", which we want to maximize. This information is returned by Voxelyze
    # in a fitness .xml file, with a tag named "NormFinalDist"
    my_objective_dict.add_objective(name="fitness", maximize=True, tag="<NormFinalDist>")
    
    my_objective_dict.add_objective(name="phenotype.instability", maximize=False, tag=None)

    # Add an objective to minimize the age of solutions: promotes diversity
    my_objective_dict.add_objective(name="age", maximize=False, tag=None)

    # Adding another objective called "num_voxels", which we want to minimize in order to minimize
    # the amount of material employed to build the robot, promoting at the same time non-trivial
    # morphologies.
    # This information can be computed in Python (it's not returned by Voxelyze, thus tag=None),
    # which is done by counting the non empty voxels (material != 0) composing the robot.
    my_objective_dict.add_objective(name="phenotype.size", maximize=False, tag=None)
    


    # Initializing a population of SoftBots
    my_pop = CellBotPopulation(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)

    # Setting up our optimization
    my_optimization = ParetoOptimization(my_sim, my_env, my_pop,
                                         mutation_func=create_new_children_through_mutation_cell)

    
    if not os.path.isfile("./" + RUN_DIR + "/pickledPops/Gen_0.pickle"):
        # start optimization
        my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                            directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                            time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                            save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES, 
                            plot_fitness_every=PLOT_FITNESS_EVERY)

    else:
        continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
                                 max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
                                 checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_POPULATION_EVERY,
                                 save_lineages=SAVE_LINEAGES, plot_fitness_every=PLOT_FITNESS_EVERY)