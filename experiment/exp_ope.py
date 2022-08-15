import numpy as np
import polars as pl
from collections import OrderedDict


users = np.array(['user1', 'user2', 'user3'])

products = np.array(
    [
        'product_a',
        'product_b',
        'product_c',
        'product_d',
        'product_e',
        'product_f',
        'product_g',
    ]
)

satisfaction = {
    'product_a': 100,
    'product_b': 150,
    'product_c': 100,
    'product_d': 200,
    'product_e': 500,
    'product_f': 120,
    'product_g': 160,
}


def will_purchase(user, product):
    if user == 'user1' and (
        product == 'product_a' or product == 'product_b' or product == 'product_c'
    ):
        return True
    elif user == 'user2' and (product == 'product_d' or product == 'product_e'):
        return True
    elif user == 'user3' and (product == 'product_f' or product == 'product_g'):
        return True
    else:
        return False


def choose_user():
    return np.random.choice(users, size=1)


def logging_policy():
    return np.random.choice(products, size=1), 1 / len(products)


class TargetPolicy:
    def __init__(self):
        self.user_probs = {
            'user1': np.array([0.1, 0.1, 0.2, 0.1, 0.15, 0.15, 0.20]),
            'user2': np.array([0.1, 0.10, 0.05, 0.25, 0.3, 0.1, 0.1]),
            'user3': np.array([0.06, 0.06, 0.3, 0.06, 0.06, 0.4, 0.06]),
        }

        for user, probs in self.user_probs.items():
            assert probs.sum() == 1
            assert len(probs) == len(products)

    def recommend(self, user):
        user_prob = self.user_probs[user]
        product = np.random.choice(products, size=1, p=user_prob)
        product_idx = np.where(products == product)
        prob = user_prob[product_idx]

        return product, prob

    def get_prob(self, user, product):
        user_prob = self.user_probs[user]
        product_idx = np.where(products == product)
        product_prob = user_prob[product_idx]

        return product_prob


def compute_satisfaction(user, product):
    if will_purchase(user, product):
        return satisfaction[product.item()]
    else:
        return 0


def create_logs(n=1000):
    logs = []
    target_policy = TargetPolicy()

    for _ in range(n):
        user = choose_user()

        logging_product, logging_prob = logging_policy()
        model_prob = target_policy.get_prob(user.item(), logging_product)

        target_product, _ = target_policy.recommend(user.item())

        logging_satisfaction = compute_satisfaction(user, logging_product)
        target_satisfaction = compute_satisfaction(user, target_product)

        log = OrderedDict(
            {
                'user_features': user.item(),
                'item_placed': logging_product.item(),
                'item_prob': logging_prob,
                'item_satisfaction': logging_satisfaction,
                'model_prob': model_prob.item(),
                'ab_test_satisfaction': target_satisfaction,
            }
        )

        logs.append(log)

    return pl.DataFrame(logs)


logs = create_logs(n=10000)
logging_policy_reward = logs['item_satisfaction'].mean()
target_policy_reward = logs['ab_test_satisfaction'].mean()

print(f'Expected reward from logging policy: {logging_policy_reward: .2f}')
print(f'Expected reward from target policy: {target_policy_reward: .2f}')


# IPS
def compute_ips(df):
    assert {'model_prob', 'item_prob', 'item_satisfaction'}.issubset(df.columns)
    return (df['model_prob'] / df['item_prob'] * df['item_satisfaction']).mean()


ips_est = compute_ips(logs)
print(f'Estimated reward from IPS: {ips_est: .2f}')


# Capped IPS
def compute_capped_ips(df, cap=10):
    capped_probs = np.minimum(df['model_prob'] / df['item_prob'], cap)
    return (capped_probs * df["item_satisfaction"]).mean()


capped_ips_est = compute_capped_ips(logs, cap=5)
print(f'Estimated reward from Capped IPS: {capped_ips_est: .2f}')


# NCIS
def compute_ncis(df, cap=10):
    capped_probs = np.minimum(df['model_prob'] / df['item_prob'], cap)
    return (capped_probs * df["item_satisfaction"]).mean() / capped_probs.mean()


ncis_est = compute_ncis(logs, cap=5)
print(f'Estimated reward from NCIS: {ncis_est: .2f}')
