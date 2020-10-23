import numpy as np
import random, pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


def updata(frame_number):
    # a = [[1,2,3],[4,5,6],[7,8,9]]
    a = np.asarray(Tracker[frame_number])
    # print (a[:, 0])

    idata['position'][:, 0] = a[:, 0]
    idata['position'][:, 1] = a[:, 1]
    scat.set_offsets(idata['position'])
    ax.set_xlabel('Latitude', fontsize = 15)
    ax.set_ylabel('Longitude', fontsize = 15)
    ax.set_title("PDR %f; Latency %f; Residual Energy %f" %
                 (a[0, 2], a[0, 3], a[0, 4]), fontsize = 20)


Xlim = [40.5, 41.0]
Ylim = [73.7, 74.2]

eG = 20
rad = 0.02
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(eG)]

fig = plt.figure(figsize = (10, 10))
ax = fig.subplots()

Tracker = pickle.load(open('Tracker.p', 'rb'))

idata = np.zeros(len(Tracker[0]), dtype = [('position', float, 2)])
ax.set_title(label='lets begin', fontdict = {'fontsize': 12}, loc = 'center')

ax.set_xlim(Xlim[0], Xlim[1])
ax.set_ylim(Ylim[0], Ylim[1])

# Place nodes
scat = ax.scatter(idata['position'][:, 0], idata['position'][:, 1], s = 10, alpha = 0.8, edgecolors='None', c = colors)

# Place edges
# plots = ax.plot(Xlim, Ylim)

line_ani = FuncAnimation(fig, updata, frames = len(Tracker), interval = 2)

Writer = animation.PillowWriter(fps = 30)
line_ani.save('viz.gif', writer = Writer)

plt.show()