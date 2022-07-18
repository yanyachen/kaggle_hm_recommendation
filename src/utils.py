import math
from tqdm import tqdm


def exp_decay_fn(time_decay_init, time_decay_strength, t):
    result = time_decay_init * math.exp(
        -1.0 * time_decay_strength * math.fabs(t)
    )
    return result


def intersection(a, b):
    return list(set(a) & set(b))


def recall(pred, label):
    return len(intersection(pred, label)) / len(label)


def cg_eval(preds, labels):
    total_recall = 0
    for user_id, pred in tqdm(preds.items()):
        label = labels[user_id]
        total_recall += recall(pred, label)
    return total_recall / len(preds)
