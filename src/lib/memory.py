from collections import deque, namedtuple
import numpy as np 

class Episode:

    def __init__(self, maxLen, nSamples, nSteps=1, gamma=1):

        self.maxLen   = maxLen
        self.nSamples = nSamples
        self.nSteps   = nSteps
        self.gamma    = gamma 
        # We want to save the cumulative reward so that we are able
        # to do some form of value-based implementation if necessary
        self.Experience = namedtuple("Experience", 
            field_names = [ "state", "action", "reward", "next_state", 
                            "done", "cumReward"])
        self.memory = deque(maxlen=maxLen)

        return
        
    def append(self, state, action, reward, next_state, done):
        # Notice that every time an experience is added, we
        # need to update the value of cumulative rewards. So
        # We shall save them as an array at the end ...

        for m in self.memory:
            m.cumReward.append((reward, next_state))

        e = self.Experience(state, action, reward, next_state, done, [])
        self.memory.append(e)

        return

    def sample(self, nSamples=None):


        if nSamples is None:
            nSamples = self.nSamples

        retVals = np.random.choice( range(len(self.memory)), nSamples )
        results = []
        for i in retVals:

            # These are the current states ...
            # ----------------------------------
            s = self.memory[i].state
            a = self.memory[i].action
            r = self.memory[i].reward

            # This is the future reward and next state ...
            # --------------------------------------------
            if self.nSteps == 1:
                ns         = self.memory[i].next_state
                cum_reward = 0
            else:

                try:
                    nr, ns     = zip(*self.memory[ i ].cumReward)
                    nSteps = min(len( nr ), self.nSteps)
                    cum_reward = 0
                    for j in range(nSteps-1):
                        cum_reward += nr[j]*self.gamma**j
                    ns = ns[ nSteps - 1 ]
                except Exception as e:
                    print('Error: {}'.format(e))
                    ns         = self.memory[i].next_state 
                    cum_reward = 0
                
            result = (s, a, r, ns, cum_reward)
            results.append(result)

        return results

class ReplayBuffer:

    def __init__(self, maxEpisodes):
        self.maxEpisodes  = maxEpisodes
        self.memory       = deque(maxlen=maxEpisodes)
        return

    def append(self, episode):
        self.memory.append(episode)

    def deleteNEpisodes(self, N=5):
        if len(self.memory) < N:
            return

        # -----------------------------------------------------------------
        # We want to be able to drop episodes that are equal to zero.
        # Over time, we want to be able to maintain a fairly healthy
        # population of good episodes that it can learn from.
        # -----------------------------------------------------------------
        episodeSum = [ sum([m[2] for m in episode.memory])  for episode in self.memory]
        episodeSum = np.array(episodeSum) + 1e-5
        episodeSum = 1/episodeSum 
        episodeSum /= episodeSum.sum()

        posToDrop = np.random.choice( range(len(self.memory)), N, replace=False, p=episodeSum  )
        posToDrop = sorted(posToDrop, reverse=True)

        for p in posToDrop:
            del self.memory[p]

        return

    def sample(self, nSamples, nEpSamples=None):
        results = []
        for episode in self.memory:
            results += episode.sample(nEpSamples)

        # Sampling depends upon the score ...
        episodeSum = np.array([r[2] for r in results])
        episodeSum = episodeSum/episodeSum.sum()

        pos     = np.random.choice(range(len(results)), nSamples, replace=False, p=episodeSum)
        results = [results[p] for p in pos]

        return results

