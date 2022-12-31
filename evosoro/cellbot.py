import numpy as np
from copy import deepcopy
import math
import torch
import cc3d

from evosoro.softbot import Phenotype, Population
from evosoro.tools.utils import stable_sigmoid, rhasattr, rsetattr

class CellBotGenotype(object):
    """A container for multiple networks, 'genetic code' copied with modification to produce offspring."""

    def __init__(self, model):

        """
        Parameters
        ----------
        model : 3-tuple (x, y, z)
            The class containing the neural network for calculating the cellular automata

        """
        self.model = model

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new
     

class CellBotPhenotype(Phenotype):
    """Physical manifestation of the genotype - determines the physiology of an individual."""

    def __init__(self, genotype, eval_stage):

        """
        Parameters
        ----------
        genotype : Genotype()
            Defines particular networks (the genome).

        eval_stage: integer
            After how many growth stages the CellBot should be evaluated.
        """
        self.genotype = genotype
        self.eval_stage = eval_stage
        self.changes_from_parent = -1
        self.initialise()
        
    def is_valid(self):
        """Phenotype is invalid if it has size 0 at the evaluation stage"""
        if self.size > 0:
            return True
        else:
            return False
   
   
    def initialise(self):
        """Grow the CellNot phenotype for the correct number of stages and calculate the size"""
        a = np.zeros(shape=self.genotype.model.robot_shape)
        np.put(a,a.size//2,1)
        self.state_history = [a]
        self.alpha_history = [a]
        self.grow(self.eval_stage-1)
        self.size = np.count_nonzero(self.eval_state)
        
        
    def grow(self, stages=1):
        """Grow the CellBot for a number of stages
        
        Parameters
        ----------
        stages : integer
            Number of stages to grow
        """
        morphogens = np.stack((self.state_history[-1],self.alpha_history[-1]))
        for stage in range(0,stages):
            morphogens = self.grow_step(morphogens)
            self.state_history.append(morphogens[0])
            self.alpha_history.append(morphogens[1])
            
    def grow_step(self, morphogens):
        """Grow the CellBot for a single stage
        
        Parameters
        ----------
        morphogens : numpy array
            The current state of the CellBot's morphology from which to grow
        """
        output_dim =  (2,3,3,3)
        padded_morphogens = np.pad(morphogens,((0,0),(1,1),(1,1),(1,1))) # Pad the robot so that a neighbourhood can be found
        neighbours = np.lib.stride_tricks.sliding_window_view(padded_morphogens, output_dim).reshape(
            math.prod(self.genotype.model.robot_shape),2,-1) # Calculate the neighbours for each cell
        cell_input = neighbours.reshape(math.prod(self.genotype.model.robot_shape),math.prod(output_dim))
        batch = torch.Tensor(cell_input)
        batch = batch.type(torch.FloatTensor)
        output, hs = self.genotype.model(batch) # Input each cell's neighbours into the model
        output = output.detach().numpy()
        
        morph_temp = np.zeros((2, math.prod(self.genotype.model.robot_shape)))
        cell_alive = np.amax(neighbours[:,1,:],axis=-1) > 0.1 # Calculate whether cells should be alive or dead
        morph_temp[0,cell_alive] = np.argmax(output[cell_alive,:-1],axis=1) # Calculate the material from the maximum of the first outputs for each cell
        morph_temp[1,cell_alive] = stable_sigmoid(output[cell_alive,-1]) # Calculate the alpha value
        morph_temp = morph_temp.reshape(2, self.genotype.model.robot_shape[0], self.genotype.model.robot_shape[1], self.genotype.model.robot_shape[2])
        
        labels_out = cc3d.largest_k(morph_temp[0]!=0, 1, connectivity=6) # Find the largest connected area
        morph_temp[0][labels_out==0] = 0
        morph_temp[1][labels_out==0] = 0 
        return deepcopy(morph_temp)
    
    
    def _get_eval_state(self):
        """Return the state of the CellBot to be evaluated"""
        return self.state_history[self.eval_stage-1]
        
    eval_state = property(fget=_get_eval_state)
        
        
class CellBot(object):
    """A CellBot is a 3D creature composed of a continuous arrangement of connected voxels with varying softness."""

    def __init__(self, max_id, objective_dict, genotype, phenotype):

        """Initialize an individual CellBot for physical simulation within VoxCad.

        Parameters
        ----------
        max_id : the lowest id tag unused
            An index to keep track of evolutionary history.

        objective_dict : ObjectiveDict()
            Defines the objectives to optimize.

        genotype : Genotype model
            Defines the model (genome).

        phenotype : Phenotype cls
            The physical manifestation of the genotype which defines an individual in simulation.

        """
        self.genotype = genotype()

        self.phenotype = phenotype(self.genotype)  # calc phenotype from genome
        
        self.id = max_id
        self.md5 = "none"
        self.dominated_by = []  # other individuals in the population that are superior according to evaluation
        self.pareto_level = 0
        self.selected = 0  # survived selection
        self.variation_type = "newly_generated"  # (from parent)
        self.parent_genotype = self.genotype  # default for randomly generated ind
        self.parent_id = -1
        self.age = 0

        # set the objectives as attributes of self (and parent)
        self.objective_dict = objective_dict
        for rank, details in objective_dict.items():
            if not rhasattr(self,details["name"]):
                rsetattr(self, details["name"], details["worst_value"])
            setattr(self, "parent_{}".format(details["name"]), details["worst_value"])

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new
    

class CellBotPopulation(Population):
    """A population of CellBots."""

    def __init__(self, objective_dict, genotype, phenotype, pop_size=30):
        Population.__init__(self, objective_dict, genotype, phenotype, pop_size)
        self.max_fitness = []
        self.mean_fitness = []


    def append(self, individuals):
        """Append a list of new individuals to the end of the population.

        Parameters
        ----------
        individuals : list of/or CellBot
            A list of individual CellBots to append or a single CellBot to append

        """
        if type(individuals) == list:
            for n in range(len(individuals)):
                if type(individuals[n]) != CellBot:
                    raise TypeError("Non-CellBot added to the population")
            self.individuals += individuals

        elif type(individuals) == CellBot:
            self.individuals += [individuals]


    def add_random_individual(self):
        """ Generate a random individual"""
        valid = False
        while not valid:
            ind = CellBot(self.max_id, self.objective_dict, self.genotype, self.phenotype)
            if ind.phenotype.is_valid():
                self.individuals.append(ind)
                self.max_id += 1
                valid = True

    def update_fitness_stats(self):
        """Calcualte the average and max fitness for the population"""
        fitness_values = []
        for ind in self:
            if ind.fitness > 0:
                fitness_values.append(ind.fitness)
        self.max_fitness.append(np.max(fitness_values))
        self.mean_fitness.append(np.mean(fitness_values))

 