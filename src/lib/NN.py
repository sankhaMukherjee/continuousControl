import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from lib import utils, memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        # self.reset_parameters()

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        # self.reset_parameters()

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:

    def __init__(self):

        self.config = json.load(open('config.json'))['Agent']

        # Generates the right action ...
        self.actor1 = Actor( **self.config['actor'] ).to(device)
        self.actor2 = Actor( **self.config['actor'] ).to(device)
        self.actOptimizer = optim.Adam( self.actor1.parameters(), lr=self.config['actorLR'] )

        # Generates the right policy ...
        self.critic1 = Critic( **self.config['critic'] ).to(device)
        self.critic2 = Critic( **self.config['critic'] ).to(device)
        self.criOptimizer = optim.Adam( self.actor1.parameters(), lr=self.config['criticLR'] )

        # Create some buffer for learning
        self.buffer = memory.ReplayBuffer(self.config['ReplayBuffer']['maxEpisodes'])

    def updateBuffer(self, env, brainName):

        self.buffer = utils.updateReplayBuffer( 
            self.buffer, env, brainName, self.actor2.forward,
            tMax      = self.config['ReplayBuffer']['episodeMaxT'], 
            gamma     = self.config['ReplayBuffer']['episodeGamma'],
            numDelete = self.config['ReplayBuffer']['episodeGamma'],)

        return

    def learn(self):

        return

    def act(self, state):

        # Take a state and convert that into an action ...
        # ------------------------------------------------
        state = torch.from_numpy(state).float().to(device)
        self.actor1.eval()
        with torch.no_grad():
            action = self.actor1(state).cpu().data.numpy()
        
        return action

    def playEpisode(self, env, brainName):

        # colllect 20 episodes
        epi  = utils.collectEpisodes(env, brainName, self.act, tMax=200, gamma=1, train_mode=True)
        
        return epi

    def transferWeights(self, tau=0.1):

        for v1, v2 in zip(self.actor1.parameters(), self.actor2.parameters()):
            v2.data.copy_( tau*v1 + (1-tau)*v2 )

        for v1, v2 in zip(self.critic1.parameters(), self.critic2.parameters()):
            v2.data.copy_( tau*v1 + (1-tau)*v2 )




