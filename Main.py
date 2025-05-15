import numpy as np
import matplotlib.pyplot as plt
import random

rates = [0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.01]
N = 10000
d = len(rates)

def ambiente(rates, N):
    d = len(rates)
    X = np.zeros((N, d))
    for i in range(N):
        for j in range(d):
            if np.random.rand() < rates[j]:
                X[i][j] += 1
    return X

def randomSelection(X):
    N, d = X.shape
    strategiesSelected = []
    totalReward = 0
    for i in range(N):
        strategy = random.randrange(d)
        reward = X[i, strategy]
        totalReward += reward
        strategiesSelected.append(strategy)
    return totalReward, strategiesSelected

def thompsonSampling(X):
    N, d = X.shape
    strategiesSelected = []
    totalReward = 0
    reward1 = [0]*d
    reward0 = [0]*d

    for i in range(N):
        select = 0
        maxRandom = 0
        for j in range(d):
            maxBeta = np.random.beta(reward1[j]+1,reward0[j]+1)
            if maxBeta > maxRandom:
                maxRandom = maxBeta
                select = j
        reward = X[i][select]

        if reward == 1:
            reward1[select] += 1
        else:
            reward0[select] += 1

        totalReward += reward
        strategiesSelected.append(select)

    return totalReward, strategiesSelected

X = ambiente(rates, N)
totalRewardRS, strategiesSelectedRS = randomSelection(X)
totalRewardTS, strategiesSelectedTS = thompsonSampling(X)

relativeReturn = (totalRewardTS - totalRewardRS) / totalRewardRS * 100
print("Thompson se saiu {:.0f}%".format(relativeReturn), 'melhor que o Random Selection')

plt.hist(strategiesSelectedTS)
plt.title('Histograma das seleções do Thompson Sampling')
plt.xlabel('Estratégias')
plt.ylabel('Número de vezes que a estratégia foi selecionada')
plt.show()
