#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:32:09 2022

@author: tree
"""
import random
import os
import dill
import numpy as np
import subprocess as sub
from glob import glob
import sys
import math
import torch
import cc3d
from copy import deepcopy



sys.path.append(os.getcwd() + "../../")
from evosoro.tools.data_analysis import get_all_data, plot_time_series
from evosoro.tools.utils import natural_sort
from evosoro.tools.logging import plot_growth
from evosoro.tools.utils import stable_sigmoid

directory = "simulation_data/stable_control_1"

pickle_idx = 0
successful_restart = False
print(sub.call("pwd", shell=True))
while not successful_restart:
    try:
        pickled_pops = glob("../simulations/" + directory + "/pickledPops/*")
        last_gen = natural_sort(pickled_pops, reverse=True)[pickle_idx]
        with open(last_gen, 'rb') as handle:
            [optimizer, random_state, numpy_random_state] = dill.load(handle)
        successful_restart = True

    except EOFError:
        # something went wrong writing the checkpoint : use previous checkpoint and redo last generation
        sub.call("touch {}/IO_ERROR_$(date +%F_%R)".format(directory), shell=True)
        pickle_idx += 1
        pass

best_bot = optimizer.pop[0]
print(best_bot.id, best_bot.fitness)
plot_growth(best_bot, optimizer.pop.gen, "../simulations/" + directory , "best_ind_before")

best_bot.phenotype.eval_stage = 7
best_bot.phenotype.initialise()
plot_growth(best_bot, optimizer.pop.gen, "../simulations/" + directory , "best_ind_eval_7")

best_bot.phenotype.eval_stage = 14
best_bot.phenotype.max_stage = 20
best_bot.phenotype.initialise()
plot_growth(best_bot, optimizer.pop.gen, "../simulations/" + directory , "best_ind_eval_14")