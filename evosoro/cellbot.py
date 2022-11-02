import operator
import numpy as np
from copy import deepcopy
import math
import torch
import cc3d

from evosoro.softbot import Phenotype, Population
from evosoro.tools.utils import stable_sigmoid, xml_format, dominates

class CellBotGenotype(object):
    """A container for multiple networks, 'genetic code' copied with modification to produce offspring."""

    def __init__(self, model, orig_size_xyz=(6, 6, 6), initial_noise=1, sigma = 0.03):

        """
        Parameters
        ----------
        orig_size_xyz : 3-tuple (x, y, z)
            Defines the original 3 dimensions for the cube of voxels corresponding to possible networks outputs. The
            maximum number of SofBot voxel components is x*y*z, a full cube.

        """
        self.model = model
        self.orig_size_xyz = orig_size_xyz
        self.initial_noise = initial_noise
        self.sigma = sigma

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

        """
        self.genotype = genotype
        self.eval_stage = eval_stage
        self.initialise()
        
    def is_valid(self):
        if self.size > 0:
            return True
        else:
            return False
   

    def _get_size(self):
        return np.count_nonzero(self.eval_state)
        
    size = property(fget=_get_size)
    
    def _get_eval_state(self):
        return self.state_history[self.eval_stage-1]
        
    eval_state = property(fget=_get_eval_state)
    
    def initialise(self):
        a = np.zeros(shape=self.genotype.model.robot_shape)
        np.put(a,a.size//2,1)
        self.state_history = [a]
        self.alpha_history = [a]
        self.grow(self.eval_stage)
        
        
    def grow(self, stages=1):
        morphogens = np.stack((self.state_history[-1],self.alpha_history[-1]))
        output_dim =  (2,3,3,3)
        for stage in range(0,stages):
            
            padded_morphogens = np.pad(morphogens,((0,0),(1,1),(1,1),(1,1)))
            neighbours = np.lib.stride_tricks.sliding_window_view(padded_morphogens, output_dim).reshape(math.prod(self.genotype.model.robot_shape),2,-1)
            #debug = morphogens.squeeze()
            cell_input = neighbours.reshape(math.prod(self.genotype.model.robot_shape),math.prod(output_dim))
            batch = torch.Tensor(cell_input)
            batch = batch.type(torch.FloatTensor)
            output, hs = self.genotype.model(batch) 
            output = output.detach().numpy()
            
            morph_temp = np.zeros((2, math.prod(self.genotype.model.robot_shape)))
            cell_alive = np.amax(neighbours[:,1,:],axis=-1) > 0.1
            morph_temp[0,cell_alive] = np.argmax(output[cell_alive,:-1],axis=1)
            morph_temp[1,cell_alive] = stable_sigmoid(output[cell_alive,-1])
            morph_temp = morph_temp.reshape(2, self.genotype.model.robot_shape[0], self.genotype.model.robot_shape[1], self.genotype.model.robot_shape[2])
            
            labels_out = cc3d.largest_k(morph_temp[0]!=0, 1, connectivity=6)
            morph_temp[0][labels_out==0] = 0
            morph_temp[1][labels_out==0] = 0        
            
            morphogens = deepcopy(morph_temp)
            self.state_history.append(morph_temp[0])
            self.alpha_history.append(morph_temp[1])
   
        

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

        genotype : Genotype cls
            Defines the networks (genome).

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
            if details["name"] != "age":
                setattr(self, details["name"], details["worst_value"])
            setattr(self, "parent_{}".format(details["name"]), details["worst_value"])

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new


class CellBotPopulation(Population):
    """A population of CellBots."""




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
        valid = False
        while not valid:
            ind = CellBot(self.max_id, self.objective_dict, self.genotype, self.phenotype)
            if ind.phenotype.is_valid():
                self.individuals.append(ind)
                self.max_id += 1
                valid = True


 