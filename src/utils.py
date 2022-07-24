import math
from tqdm import tqdm
import numpy as np
import faiss


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


class InnerProductSearcher(object):
    def __init__(self, embeddings, embedding_size, labels):
        self.index = faiss.IndexFlatIP(embedding_size)
        self.index.add(embeddings)
        self.labels = labels

    def search(self, inputs, k):
        if len(inputs.shape) == 1:
            inputs = np.expand_dims(inputs, axis=0)
        distances, indices = self.index.search(inputs, k)
        result = [
            [self.labels[i] for i in each]
            for each in indices
        ]
        return result
