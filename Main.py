import numpy as np
import matplotlib.pyplot as plt
import random

# Parâmetros
rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
N = 10000
d = len(rates)


def simulate_environment(rates, N):
    d = len(rates)
    X = np.zeros((N, d))
    for i in range(N):
        for j in range(d):
            if np.random.rand() < rates[j]:
                X[i][j] = 1
    return X


def random_selection(X):
    N, d = X.shape
    strategies_selected = []
    total_reward = 0
    for n in range(N):
        strategy = random.randrange(d)
        reward = X[n, strategy]
        total_reward += reward
        strategies_selected.append(strategy)
    return total_reward, strategies_selected


def thompson_sampling(X):
    N, d = X.shape
    strategies_selected = []
    total_reward = 0
    numbers_of_rewards_1 = [0] * d
    numbers_of_rewards_0 = [0] * d

    for n in range(N):
        max_random = 0
        strategy = 0
        for i in range(d):
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
            if random_beta > max_random:
                max_random = random_beta
                strategy = i
        reward = X[n, strategy]

        if reward == 1:
            numbers_of_rewards_1[strategy] += 1
        else:
            numbers_of_rewards_0[strategy] += 1

        total_reward += reward
        strategies_selected.append(strategy)

    return total_reward, strategies_selected


X = simulate_environment(rates, N)
total_reward_rs, strategies_selected_rs = random_selection(X)
total_reward_ts, strategies_selected_ts = thompson_sampling(X)


relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100
print("Relative Return: {:.0f} %".format(relative_return))


plt.hist(strategies_selected_ts)
plt.title('Histogram of selections of Thompson Sampling')
plt.xlabel('Strategy')
plt.ylabel('Number of times the strategy was selected')
plt.show()