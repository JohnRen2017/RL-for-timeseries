import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Environ:
    def __init__(self, data=None, max_iteration=None, lookback=3):
        if data is not None:
            if isinstance(data, pd.DataFrame):
                data = data.values
            elif isinstance(data, np.ndarray):
                data = data
            self.data = self.normalize(data)
        else:
            self.data = self.gendata()

        if max_iteration is not None:
            self.max_iter = min(max_iteration, self.data.shape[0])
        else:
            self.max_iter = self.data.shape[0]

        self.lookback = lookback
        self.done = False

    @staticmethod
    def gendata():
        _points = 24 * 30
        _x = np.arange(_points)
        _data = np.zeros((_points, 3))
        _data[:, 0] = abs(np.sin(_x * (np.pi/48)))
        _data[:, 1] = abs(np.cos(_x * (np.pi/48)))
        _data[:, 2] = abs(np.sin(_x * (np.pi/24)))
        return _data

    def reset(self):
        self.iteration = 0
        self.next_state = self.data[self.iteration: self.iteration + self.lookback, :]
        self.pred_states = []
        self.true_states = []
        return self.next_state

    def normalize(self, data):
        temp = data[:, 0] # only normalize rrc
        self.rrcmin = np.min(temp, axis=0)
        self.rrcmax = np.max(temp, axis=0)
        temp = (temp - self.rrcmin) / (self.rrcmax - self.rrcmin)
        data[:, 0] = temp
        return data

    @staticmethod
    def _rewardcalc(pred, true):
        return 1 / np.sqrt(np.sum(np.square(pred - true)))

    def step(self, action):
        """
        action: a list with three items;
                each item is a continous value in the range of [0, 1];
                action represents final result--deterministic policy;
        """
        action = list(action)
        true_state = self.data[self.lookback + self.iteration, :]
        self.true_states.append(true_state)
        reward = self._rewardcalc(action, true_state)
        self.pred_states.append(action)
        self.iteration += 1
        self.next_state = self.data[self.iteration: self.iteration + self.lookback, :]
        if self.iteration >= self.max_iter:
            self.done = True
        return self.next_state, reward, self.done
    
    def plotdata(self):
        rows = self.data.shape[1]
        length = self.data.shape[0]
        _, axs = plt.subplots(nrows=rows, ncols=1, sharex=True, figsize=(10, 4*rows))
        for i in range(rows):
            axs[i].plot(range(length), self.data[:, i])
            axs[i].set_ylabel(f'NO.{i+1}')
        axs[0].set_title('Three KPIs vs. time')
        axs[i].set_xlabel('time steps')
        plt.show()

    def render(self):
        pass

if __name__ == "__main__":
    # POINTS = 1000
    # x = np.linspace(0, 8*np.pi, num=POINTS)
    # data = pd.DataFrame(np.zeros((POINTS, 3)))
    # data.iloc[:, 0] = np.cos(x)
    # data.iloc[:, 1] = np.sin(x)
    # data.iloc[:, 2] = np.cos(2*x)
    # print("dataset head is:\n")
    # print(data.head())

    env = Environ(data=None, max_iteration=2, lookback=3)
    obv = env.reset()
    print("initial observation is:\n")
    print(obv)

    # env.plotdata()

    s = 0
    while not env.done:
        s += 1
        print("***iteration step: {}***".format(s))
        next_state, reward, done = env.step([0.9, 0.05, 0.9])
        print("next_state is:")
        print(next_state)
        print("reward is :{}".format(reward))
        print("done value is: {}".format(done))
