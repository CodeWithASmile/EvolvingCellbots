
import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils import parameters_to_vector, vector_to_parameters

class CA(nn.Module):
    
    def __init__(self, robot_shape, hidden_dim=64, output_dim = 6, control=False, initial_noise = 1,
                 sigma = 0.03, allow_neutral_mutations=False):
        super(CA, self).__init__()
        self.output_dim = output_dim
        self.robot_shape = robot_shape
        self.initial_noise = initial_noise
        self.sigma = sigma
        self.allow_neutral_mutations = allow_neutral_mutations
        
        self.input_dim = 3**len(self.robot_shape)*2
        if control:
            self.input_dim +=1

        self.linear1 = nn.Linear(self.input_dim, 64) 
        self.linear2 = nn.Linear(64, 64) 
        self.linear3 = nn.Linear(64, self.output_dim)
        
        self.weight_shape = parameters_to_vector(self.parameters() ).detach().numpy().shape


    def forward(self,x,hidden=None):

        x = torch.FloatTensor(x)/3.0
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x= self.linear3(x)
        #x = F.log_softmax(x)   
        return x, hidden

    def get_weights(self):
        return  parameters_to_vector(self.parameters() ).detach().numpy()
  
    def set_weights(self, w):
        vector_to_parameters(w,self.parameters())
      
    def initialise(self):
        w = torch.from_numpy( np.random.normal(0, 1, self.weight_shape)*0.1*self.initial_noise).float()
        self.set_weights(w)
      
    def mutate(self):
        w = parameters_to_vector(self.parameters() ).detach()
        new = w + torch.from_numpy( np.random.normal(0, 1, self.weight_shape) * self.sigma).float()
        self.set_weights(new)
  
