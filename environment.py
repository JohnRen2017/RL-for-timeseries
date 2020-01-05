import numpy as np
import pandas as pd

class Environ:
    def __init__(self, data, max_iteration=None):
        assert isinstance(data, pd.DataFrame)
        self.data = self.normalize(data.values)
        self.max_iter = max_iteration
    def reset(self):
        self.iteration = 0
        self.next_state = self.data[self.iteration: self.iteration+3, :]
        self.pred_states = []
        self.true_states = []
        return self.next_state
    def normalize(self, data):
        temp = data[:, 0]
        self.rrcmin = np.min(temp, axis=0)
        self.rrcmax = np.max(temp, axis=0)
        temp = (temp - self.rrcmin) / (self.rrcmax - self.rrcmin)
        data[:, 0] = temp
        return data
    @staticmethod
    def rewardcalc(pred, true):
        return 1 / np.sqrt(np.sum(np.square(pred - true)))
    def step(self, action):
        """
        action: a list with three items;
                each item is a continous value in the range of [0, 1];
                action represents final result--deterministic policy;
        """
        action = list(action)
        done = False
        true_state = self.data[3+self.iteration, :]
        self.true_states.append(true_state)
        reward = self.rewardcalc(action, true_state)
        self.pred_states.append(action)
        self.iteration += 1
        self.next_state = self.data[self.iteration: self.iteration+3, :]
        if self.iteration == self.max_iter:
            done = False
        return self.next_state, reward, done

if __name__ == "__main__":
    POINTS = 1000
    x = np.linspace(0, 8*np.pi, num=POINTS)
    data = pd.DataFrame(np.zeros((POINTS, 3)))
    data.iloc[:, 0] = np.cos(x)
    data.iloc[:, 1] = np.sin(x)
    data.iloc[:, 2] = np.cos(2*x)
    print("dataset head is:\n")
    print(data.head())
    env = Environ(data)
    obv = env.reset()
    print("initial observation is:\n")
    print(obv)
    for s in range(5):
        print("***iteration step: {}***".format(s+1))
        next_state, reward, done = env.step([0.9, 0.05, 0.9])
        print("next_state is:")
        print(next_state)
        print("reward is :{}".format(reward))
        print("done value is: {}".format(done))
