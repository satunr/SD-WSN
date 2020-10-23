import pickle
import numpy as np

import matplotlib.pyplot as plt

P1 = pickle.load(open('P1.p', 'rb'))
P9 = pickle.load(open('P9.p', 'rb'))

T = len(P9[0])

Y1 = [np.mean([P1[i][t]for i in range(5)]) for t in range(T)]
Y1s = [np.std([P1[i][t]for i in range(5)]) for t in range(T)]

Y9 = [np.mean([P9[i][t]for i in range(5)]) for t in range(T)]
Y9s = [np.std([P9[i][t]for i in range(5)]) for t in range(T)]


plt.plot([t for t in range(T)], Y1, label = 'High Periodicity', color = 'green')
plt.fill_between([t for t in range(T)], np.array(Y1) + np.array(Y1s) / 2.0,
                 np.array(Y1) - np.array(Y1s) / 2.0, alpha = 0.3, color = 'green')

plt.plot([t for t in range(T)], Y9, label = 'Low Periodicity', color = 'brown')
plt.fill_between([t for t in range(T)], np.array(Y9) + np.array(Y9s) / 2.0,
                 np.array(Y9) - np.array(Y9s) / 2.0, alpha = 0.3, color = 'brown')

plt.legend(loc = 'lower right')

plt.xlabel('Time', fontsize = 15)
plt.ylabel('Average Number of Hops', fontsize = 15)

plt.tight_layout()
plt.savefig('Hops.png', dpi = 300)

plt.show()



