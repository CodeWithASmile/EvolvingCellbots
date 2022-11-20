import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils import parameters_to_vector, vector_to_parameters

class CA(nn.Module):
    
    def __init__(self, robot_shape, hidden_dim=64, output_dim = 6, control=False, initial_noise = 1,
                 sigma = 0.03, allow_neutral_mutations=True):
        """ Extends the neural network class represent the rules of the cellular automata"""
        super(CA, self).__init__()
        
        """
        Parameters
        ----------
        robot_shape: 3-tuple (x, y, z)
            Defines the 3 dimensions of the robot.
        hidden_dim: integer
            The dimension of the hidden layer of the neural network.
        output_dim: integer
            The dimension of the output of the neural network.
        control: bool
            Whether or not a control signal is being used.
        initial_noise: float
            The noise used to generate the weights of a random individual.
        sigma: float
            The value used to calculate the mutation of the weights.
        allow_neutral_mutations:
            When mutating an individual, should weight changes that do not affect the morphology be allowed.
        """
        self.output_dim = output_dim
        self.robot_shape = robot_shape
        self.initial_noise = initial_noise
        self.sigma = sigma
        self.allow_neutral_mutations = allow_neutral_mutations
        
        # Calculate the input dimensions
        self.input_dim = 3**len(self.robot_shape)*2
        if control:
            self.input_dim +=1

        # Create the layers of the neural network
        self.linear1 = nn.Linear(self.input_dim, 64)
        self.linear2 = nn.Linear(64, 64) 
        self.linear3 = nn.Linear(64, self.output_dim)
        
        self.weight_shape = parameters_to_vector(self.parameters() ).detach().numpy().shape


    def forward(self,x,hidden=None):
        """Override the forward method of the neural network

        Parameters
        ----------
        x: tensor
            The inputs to the neural network
        """
        
        x = torch.FloatTensor(x)/3.0
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x= self.linear3(x)

        return x, hidden

    def get_weights(self):
        """Return the weights of the neural network as a numpy array"""
        return  parameters_to_vector(self.parameters() ).detach().numpy()
  
    def set_weights(self, w):
        """Set the weights of the neural network from a numpy array"""
        vector_to_parameters(w,self.parameters())
      
    def initialise(self):
        """Random generate weights for the neural network"""
        w = torch.from_numpy( np.random.normal(0, 1, self.weight_shape)*0.1*self.initial_noise).float()
        self.set_weights(w)
      
    def mutate(self):
        """Mutate the weights of the neural network"""
        w = parameters_to_vector(self.parameters() ).detach()
        new = w + torch.from_numpy( np.random.normal(0, 1, self.weight_shape) * self.sigma).float()
        self.set_weights(new)
  
