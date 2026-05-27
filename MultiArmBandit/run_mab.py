
from MultiArmBandit.epsilon_greedy import EpsilonGreedy
from MultiArmBandit.decay_epsilon_greedy import DecayEpsilonGreedy
from MultiArmBandit.ucb import UCB
from MultiArmBandit.thomson_sampling import ThomsonSampling
from environment.mab_env import MABEnv
import matplotlib.pyplot as plt


def plot_results(solvers, solver_names):

    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        # For plotting total number of regrets
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regrets")
    plt.title("%d-Armed Bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    K = 10  # Total number of arms
    T = 5000  # Total number of time stpes
    mab_env = MABEnv(K=K, seed=100)

    # Epsilon-Greedy solvers
    epsilon_greedy_solvers = EpsilonGreedy(bandit=mab_env, epsilon=0.1)
    epsilon_greedy_solvers.run(T)

    decay_epsilon_greedy_solvers = DecayEpsilonGreedy(
        bandit=mab_env, init_epsilon=0.1, decay_rate=0.99)
    decay_epsilon_greedy_solvers.run(T)

    ucb_solvers = UCB(bandit=mab_env, coef=1)
    ucb_solvers.run(T)

    thomson_sampling_solvers = ThomsonSampling(mab_env)
    thomson_sampling_solvers.run(T)

    all_solvers = [epsilon_greedy_solvers,
                   decay_epsilon_greedy_solvers, ucb_solvers, thomson_sampling_solvers]
    all_solver_names = ['Epsilon-Greedy',
                        "Decay Epsilon-Greedy ", "UCB", "Thomson Sampling"]
    plot_results(solvers=all_solvers, solver_names=all_solver_names)
