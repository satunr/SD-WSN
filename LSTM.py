import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.neighbors import NearestCentroid


# X = [[0 for i in range(10 - len([int(char) for char in bin(i)[2:]]))] + [int(char) for char in bin(i)[2:]] for i in range(1000)]
# Y = []
# y = [np.random.choice([6 * (i + 1), random.randint(0, 10)], p = [1.0, 0.0], size = 1)[0] for i in range(4)]
#
# for i in range(250):
#     Y.extend(y)
#
# print (X)
# print (Y)
#
# clf = tree.DecisionTreeClassifier()
# clf = NearestCentroid()
# clf = clf.fit(X, Y)
# print (clf.predict([[100]]))
#
# # Supervised Neural Network Model
# clf = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes = (5, 5, 5, 5), random_state = 1)
# clf = clf.fit(X, Y)
# scores = cross_val_score(clf, X, Y, cv = 500)
# print (scores)
# print (clf.predict([[1, 1, 1, 1, 1, 0, 1, 1, 0, 0]]))


def make_binary(t):
    return [0 for t in range(10 - len([int(char) for char in bin(t)[2:]]))] + [int(char) for char in bin(t)[2:]]


N = 10
T = 100
R = [0.01 for _ in range(5)] + [0.2 for _ in range(3)] + [0.6 for _ in range(2)]
start_plot = 10

clf = NearestCentroid()

# Moves
D = {r: [] for r in range(N)}

# Accuracy
A = {r: [0, 0] for r in range(N)}

# Accuracy plot
AP = {r: [0 for _ in range(start_plot)] for r in range(N)}

for t in range(T):

    for r in range(N):
        # Predict
        val = None
        if t > start_plot:
            x = [make_binary(i) for i in range(len(D[r]))]
            y = [D[r][i] for i in range(len(D[r]))]

            clf.fit(x, y)
            val = clf.predict([make_binary(t)])
            # print (val)

        # Learn
        p = np.random.choice([t % 4, random.randint(0, 10)], p = [1.0 - R[r], R[r]], size = 1)
        p = p[0]
        D[r].append(p)

        if t > start_plot:
            # Update accuracy
            A[r][1] += 1
            if val[0] == p:
                A[r][0] += 1

            AP[r].append(float(A[r][0])/float(A[r][1]))
print (A)

# Plot accuracy
for r in range(N):
    if r < 5:
        plt.plot([i for i in range(len(AP[r]))], AP[r], label = 'Node' + str(r))
    elif r < 8:
        plt.plot([i for i in range(len(AP[r]))], AP[r], label = 'Node' + str(r), linestyle = 'dashed')
    else:
        plt.plot([i for i in range(len(AP[r]))], AP[r], label = 'Node' + str(r), linestyle = 'dotted')

# plt.legend()
plt.xlabel('Time', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)

plt.tight_layout()
plt.savefig('NC.png', dpi = 300)
plt.show()



