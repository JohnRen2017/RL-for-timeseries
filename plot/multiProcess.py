import os
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append("..\RL4TimeSeries")
from environment import Environ
from multiprocessing import Process


class SPEnv(Environ):
    def __init__(self, data=None, max_iteration=10, lookback=3, genlength=24 * 30):
        super().__init__(
            data=data,
            max_iteration=max_iteration,
            lookback=lookback,
            genlength=genlength,
        )
        # self.FPATH = r"C:\PythonCode\myRL\RL4TimeSeries"
        self.FPATH = os.getcwd()

    def resetfpath(self, path):
        self.FPATH = path

    def getcontent(self, path):
        with open(os.path.join(self.FPATH, path), "r") as f:
            lines = f.readlines()[::2]
            vals = [list(line.split(",")[:-1]) for line in lines]
        return [[float(i) for i in val] for val in vals]

    def render(self):
        global true, pred, axes
        figure, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))
        true = np.array(self.getcontent("true.txt"))
        pred = np.array(self.getcontent("pred.txt"))

        def animate(i):
            for j in range(true.shape[1]):
                axes[j].cla()
                if j == 0:
                    axes[j].set_title("Comparison between true and predict values")
                axes[j].plot(
                    list(range(true.shape[0])),
                    true[:, j],
                    color="r",
                    label="true value_{}".format(j + 1),
                )
                axes[j].scatter(
                    list(range(pred.shape[0])),
                    pred[:, j],
                    color="b",
                    s=np.abs(true[:, j] - pred[:, j]) * 5,
                    marker="D",
                    label="pred value_{}".format(j + 1),
                )
                axes[j].set_ylabel("KPI-{}".format(j + 1))
                axes[j].legend(loc="center left")
            axes[j].set_xlabel("time")

        ani = FuncAnimation(figure, animate, interval=1000)
        plt.tight_layout(pad=1.6)
        plt.show()


if __name__ == "__main__":
    myenv = SPEnv(data=None, max_iteration=10, lookback=3, genlength=24 * 30)
    obv = myenv.reset()
    print("initial observation is:\n")
    print(obv)

    s = 0
    while not myenv.done:
        if s == 2:
            # myenv.render()
            p = Process(target=myenv.render)
            p.start()
        s += 1
        print("********iteration step: {}********".format(s))
        next_state, reward, done = myenv.step([0.9, 0.05, 0.9])
        print("next_state is:")
        print(next_state)
        print()
        print("reward is :{}".format(reward))
        print("done value is: {}".format(done))
        time.sleep(1)
