import logging
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)



def get_weights_path(folder, gen, individual_id):
    return "{0}/weights_gen_{1:03d}_id_{2:03d}.pt".format(folder,gen,individual_id)

def get_morphogens_path(folder, gen, individual_id):
    return "{0}/morphogens_gen_{1:03d}_id_{2:03d}".format(folder,gen,individual_id)

def get_dev_states_path(folder, gen, individual_id):
    return "{0}/dev_states_gen_{1}_id_{2}".format(folder,gen,individual_id)

def get_alpha_path(folder, gen, individual_id):
    return "{0}/alpha_gen_{1:03d}_id_{2:03d}".format(folder,gen,individual_id)


def plot_growth(ind, gen, run_directory, run_name):
    fig = plt.figure(figsize=(20,10))
    for it in range(0,len(ind.phenotype.state_history)):
        #print(it)
        voxels = ind.phenotype.state_history[it]
        #voxels = voxels.transpose((2,1,0))
        alpha_temp = ind.phenotype.alpha_history[it]
        #alpha_temp = alpha_temp.transpose((2,1,0))
        #print(voxels[1],voxels[1,3])
        
        ax = fig.add_subplot(4, 5, it+1, projection= '3d')
        #plt.subplot(3, iterations/3+12, it+1)#,figsize=(15,15))
        col = [[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0],[0, 1, 0]]

        face_col = np.concatenate( (np.array(col)[voxels.astype(int)], np.expand_dims(alpha_temp, axis=3) ) , axis=3)
        #face_col = np.concatenate( (np.array(col)[morphocegens[0].astype(int)], np.expand_dims(morphogens[1], axis=3) ) , axis=3) 
        #face_col = face_col.transpose((1,2,0))
        ax.set_aspect(aspect='auto')
        ax.voxels( voxels, facecolors=face_col,edgecolor='k')#np.array(col)[morphogens[0].astype(int)])#,

    plt.savefig(run_directory + "/bestSoFar/fitOnly/" + run_name +
    "--Gen_%04i--fit_%.08f--id_%05i.pdf" %
    (gen, ind.fitness, ind.id))
    plt.close()
    
def save_epoch_data(experiment, data, epoch):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) 
    #ax.plot(gen, maxgenfit, linestyle='--', color='b', label='maxgenfit')
    ax.plot(data['gen'], data['meanfit'], linestyle='-', color='r', label='Mean')
    ax.plot(data['gen'], data['maxfit'], linestyle='dotted', color = 'b', label='Max fitness')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.legend(loc='best')
    plt.savefig('{0}/gen_{1:03d}.pdf'.format(experiment.run_dir,epoch))
    plt.close()

    #print(population)'[
    #torch.save(data['population'], "{0}/weights_gen_{1:03d}.pt".format(experiment.folders['model_state_files'],epoch))

    str_ = str(data['fitness_log'])
    str_1 = str(data['fitness_all'])
    with open("{0}/fitness_log_gen_{1:03d}.txt".format(experiment.folders['fitness_files'], epoch), 'wt') as f:
        f.write(str_)
    with open("{0}/fitness_all_gen_{1:03d}.txt".format(experiment.folders['fitness_files'], epoch), 'wt') as g:
        g.write(str_1)

   