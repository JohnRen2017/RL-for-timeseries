import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(i):
    true = np.array(getcontent("true.txt"))
    pred = np.array(getcontent("pred.txt"))

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
            s=np.abs(true[:, j] - pred[:, j]) * 10,
            marker="D",
            label="pred value_{}".format(j + 1),
        )
        axes[j].set_ylabel("KPI-{}".format(j + 1))
        axes[j].legend(loc="center left")
    axes[j].set_xlabel("time")


def getcontent(path):
    with open(os.path.join(FPATH, path), "r") as f:
        lines = f.readlines()[::2]
        vals = [list(line.split(",")[:-1]) for line in lines]
    return [[float(i) for i in val] for val in vals]


if __name__ == "__main__":
    FPATH = r"C:\PythonCode\myRL\RL4TimeSeries"
    figure, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))
    ani = FuncAnimation(figure, animate, interval=1000)
    plt.tight_layout(pad=1.6)
    plt.show()
