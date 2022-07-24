import fileinput
import random
import time
import numpy as np


def read_data(filenames):
    """
    Reads a stream of events from the list of given files.

    Parameters
    ----------
    filenames : list
        List of filenames

    Stores
    -------
    articles : [article_ids]
    features : [[article_1_features] .. [article_n_features]]
    events : [
                 0 : displayed_article_index (relative to the pool),
                 1 : user_click,
                 2 : [user_features],
                 3 : [pool_indexes]
             ]
    """

    global articles, features, events, n_arms, n_events
    articles = []
    features = []
    events = []

    skipped = 0

    with fileinput.input(files=filenames) as f:
        for line in f:
            cols = line.split()
            if (len(cols) - 10) % 7 != 0:
                skipped += 1
            else:
                pool_idx = []
                pool_ids = []
                for i in range(10, len(cols) - 6, 7):
                    id = cols[i][1:]
                    if id not in articles:
                        articles.append(id)
                        features.append([float(x[2:]) for x in cols[i + 1: i + 7]])
                    pool_idx.append(articles.index(id))
                    pool_ids.append(id)

                events.append(
                    [
                        pool_ids.index(cols[1]),
                        int(cols[2]),
                        [float(x[2:]) for x in cols[4:10]],
                        pool_idx,
                    ]
                )
    features = np.array(features)
    n_arms = len(articles)
    n_events = len(events)
    print(n_events, "events with", n_arms, "articles")
    if skipped != 0:
        print("Skipped events:", skipped)


def evaluate(A, size=100, learn_ratio=0.9):
    """
    Policy evaluator as described in the paper
    Parameters
    ----------
    A : class (algorithm)
    size : number (run the evaluation only on a portion of the dataset)
    learn_ratio : number (perform learning(update parameters) only on a small portion of the traffic)
    Returns
    -------
    learn : array (contains the ctr for each trial for the learning bucket)
    deploy : array (contains the ctr for each trial for the deployment bucket)
    """

    start = time.time()
    # we initialize the payoff and events parameters separately for learning phase of the events and deployment phase of events.
    Payoff_deploy = 0 # total payoff for the deployment bucket
    Payoff_learn = 0  # total payoff for the learning bucket
    Events_deploy = 1 # counter of valid events for the deployment bucket
    Events_learn = 0  # counter of valid events for the learning bucket

    learn = []
    deploy = []
    global events
    if size != 100:
        k = int(n_events * size / 100)
        events = random.sample(events, k)

    """
    we run through the logged events, and treat each event either for learning & updating the parameters,
    or for deployment purposes wherein we use the reward obtained as evaluation metric
    """
    for t, event in enumerate(events):

        displayed = event[0]
        reward = event[1]
        user = event[2]
        pool_idx = event[3]

        # select the arm based on the bandit policy
        chosen = A.choose_arm(Payoff_learn + Payoff_deploy, user, pool_idx)

        """
        If, given the current history ht−1, it happens that the policy A chooses the same arm a
        as the one that was selected by the logging policy, then the event is retained
        (that is, added to the history), and the total payoff updated.
        Otherwise, if the policy A selects a different arm from the one that was taken by the logging policy,
        then the event is entirely ignored, and the algorithm proceeds to the next event without any change in its state.
        """
        if chosen == displayed:
            if random.random() < learn_ratio:
                Payoff_learn += reward
                Events_learn += 1
                A.update(displayed, reward, user, pool_idx)
                learn.append(Payoff_learn / Events_learn)
            else:
                Payoff_deploy += reward
                Events_deploy += 1
                deploy.append(Payoff_deploy / Events_deploy)

    end = time.time()

    execution_time = round(end - start, 1)
    execution_time = (
        str(round(execution_time / 60, 1)) + "m"
        if execution_time > 60
        else str(execution_time) + "s"
    )
    print(
        "{:<20}{:<10}{}".format(
            A.algorithm,
            round(Payoff_deploy / Events_deploy, 4),
            execution_time
        )
    )

    return learn, deploy


class Egreedy:
    """
    Epsilon greedy algorithm implementation
    """

    def __init__(self, epsilon):
        """
        Parameters
        ----------
        epsilon : number (Egreedy parameter, ideally between 0 and 1)
        """

        self.e = round(epsilon, 1)  # epsilon parameter for Egreedy 
        self.algorithm = "Egreedy (ε=" + str(self.e) + ")"
        self.q = np.zeros(n_arms)  # average reward for each arm -- this represents the known mean reward for each arm
        self.n = np.zeros(n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number (number of trial)
        user : array (user features)
        pool_idx : array of indexes (pool indexes for article identification)
        """

        p = np.random.rand()
        if p > self.e:
            # exploit
            return np.argmax(self.q[pool_idx])
        else:
            # explore
            return np.random.randint(low=0, high=len(pool_idx))

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices)
        Parameters
        ----------
        displayed : index (displayed article index relative to the pool)
        reward : binary (user clicked or not)
        user : array (user features)
        pool_idx : array of indexes (pool indexes for article identification)
        """

        a = pool_idx[displayed]

        # update counts pulled for chosen arm
        self.n[a] += 1

        # update average/mean value/reward for chosen arm
        self.q[a] += (reward - self.q[a]) / self.n[a]
        """
        this can also be written as:
        value = self.q[a]
        new_value = ((self.n[a]-1)/float(self.n[a])) * value + (1 / float(self.n[a])) * reward
        self.q[a] = new_value
        """


class Ucb1:
    def __init__(self, alpha):
        """
        Parameters
        ----------
        alpha : number (ucb parameter)
        """

        self.alpha = round(alpha, 1)
        self.algorithm = "UCB1 (α=" + str(self.alpha) + ")"

        self.q = np.zeros(n_arms)  # average reward for each arm
        self.n = np.ones(n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number (number of trial)
        user : array (user features)
        pool_idx : array of indexes (pool indexes for article identification)
        """

        ucb_array = (
            self.q[pool_idx] +
            np.sqrt(self.alpha * np.log(t+1) / self.n[pool_idx])
        )
        selection = np.argmax(ucb_array)
        return selection

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices)
        Parameters
        ----------
        displayed : index (displayed article index relative to the pool)
        reward : binary (user clicked or not)
        user : array (user features)
        pool_idx : array of indexes (pool indexes for article identification)
        """

        a = pool_idx[displayed]
        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]


def repeat_exp(fn, times):
    history = []
    for _ in range(times):
        algo = fn()
        _, deploy = evaluate(algo)
        rnd_ctr = deploy[-1]
        history.append(rnd_ctr)
    history = np.array(history)
    return history.mean(), history.std()


# Experiment
read_data('./temp/data1.txt')


exp_result = {
    alpha: repeat_exp(lambda: Ucb1(alpha), 100)
    for alpha in np.linspace(0, 2, 21)
}

'''
{
    0.0: (0.020576234955475448, 0.005886137744823969),
    0.1: (0.030507531333765266, 0.007650271837393907),
    0.2: (0.028563185698956172, 0.007249866854986926),
    0.3: (0.027645744845719702, 0.008345846513005668),
    0.4: (0.028740098555307840, 0.007581008369611694),
    0.5: (0.027346290028847805, 0.007275630404954568),
    0.6: (0.028029038820398190, 0.007218069997984095),
    0.7: (0.027587391876473120, 0.006951837877557924),
    0.8: (0.028382804989856472, 0.00810094562624723),
    0.9: (0.027846687708247450, 0.007394350689299963),
    1.0: (0.027320182692765200, 0.007539344447758061),
    1.1: (0.027550148363230727, 0.007285124802075134),
    1.2: (0.026480516291572612, 0.007463873896617457),
    1.3: (0.026745237444120294, 0.008496550966999247),
    1.4: (0.027127600966350710, 0.006452810255943557),
    1.5: (0.028688462339344390, 0.007434862545508341),
    1.6: (0.026617031544399710, 0.007188936121614876),
    1.7: (0.027534789517639657, 0.006939645941241717),
    1.8: (0.026263164417616433, 0.007425634057497431),
    1.9: (0.025973250750318283, 0.007382949547938601),
    2.0: (0.026615860477799810, 0.007434003101466481)
}
'''



