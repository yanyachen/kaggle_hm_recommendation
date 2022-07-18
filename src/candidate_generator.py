from collections import (
    Counter, defaultdict
)
import math
import numpy as np
import pandas as pd
from tqdm import tqdm


class ItemCF:
    def __init__(
        self,
        user_id_colname, item_id_colname, time_colname
    ):
        # Constant
        self.user_id_colname = user_id_colname
        self.item_id_colname = item_id_colname
        self.time_colname = time_colname

        # Data
        self.user_history = defaultdict(
            lambda: {'num_item': 0, 'item_id': [], 'time': []}
        )
        self.item_counter = Counter()
        self.item_item_cooccurrence = defaultdict(lambda: defaultdict(float))

    def train(
        self, df,
        positive_time_strength, negative_time_strength,
        i2i_index_decay_fn, i2i_time_decay_fn,
        user_frequency_regularization_fn,
        item_frequency_regularization_fn
    ):
        # Time Sorting
        df.sort_values(by=self.time_colname, axis=0, ascending=True, inplace=True)

        # Data Processing
        for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
            user_id = getattr(row, self.user_id_colname)
            item_id = getattr(row, self.item_id_colname)
            time = getattr(row, self.time_colname)
            self.user_history[user_id]['num_item'] += 1
            self.user_history[user_id]['item_id'].append(item_id)
            self.user_history[user_id]['time'].append(time)
            self.item_counter[item_id] += 1

        # Update Cooccurrence
        for user_id, item_interaction_history in tqdm(self.user_history.items()):
            user_interaction_num_item = item_interaction_history['num_item']
            user_interaction_item_id_list = item_interaction_history['item_id']
            user_interaction_time_list = item_interaction_history['time']

            for i in range(0, user_interaction_num_item, +1):
                for j in range(0, i, +1):
                    current_item_id = user_interaction_item_id_list[i]
                    current_time = user_interaction_time_list[i]
                    history_item_id = user_interaction_item_id_list[j]
                    history_time = user_interaction_time_list[j]

                    # Inverse User Frequency
                    user_frequency_weight = user_frequency_regularization_fn(
                        user_interaction_num_item
                    )

                    # Positive
                    self.item_item_cooccurrence[history_item_id][current_item_id] += \
                        1.0 * positive_time_strength * \
                        i2i_index_decay_fn(i -j) * \
                        i2i_time_decay_fn(current_time - history_time) * \
                        user_frequency_weight
                    # Negative
                    self.item_item_cooccurrence[current_item_id][history_item_id] += \
                        1.0 * negative_time_strength * \
                        i2i_index_decay_fn(i - j) * \
                        i2i_time_decay_fn(current_time - history_time) * \
                        user_frequency_weight

        # Inverse Item Frequency
        for item_i in tqdm(self.item_item_cooccurrence.keys()):
            for item_j in self.item_item_cooccurrence[item_i]:
                self.item_item_cooccurrence[item_i][item_j] /= \
                    item_frequency_regularization_fn(
                        self.item_counter[item_i],
                        self.item_counter[item_j]
                    )

    def predict_each_user(
        self,
        user_id, time,
        u2i_index_decay_fn, u2i_time_decay_fn,
        last_n, top_k, filtered
    ):
        # Scoring
        itemcf_item_score_dict = defaultdict(float)
        num_triggered_item = min(self.user_history[user_id]['num_item'], last_n)
        interacted_item_id_set = set(self.user_history[user_id]['item_id'])
        for i in range(1, num_triggered_item + 1, +1):
            # Each Interacted Item
            history_item_id = self.user_history[user_id]['item_id'][-i]
            history_time = self.user_history[user_id]['time'][-i]
            # User to Item Relevance based on Time Decay
            user_relevance_to_history_item_id = u2i_index_decay_fn(i) * \
                u2i_time_decay_fn(time - history_time)
            # Compute Item Score
            for recommended_item_id, cooccurrence_score in self.item_item_cooccurrence[history_item_id].items():
                if (not filtered) or (recommended_item_id not in interacted_item_id_set):
                    itemcf_item_score_dict[recommended_item_id] += user_relevance_to_history_item_id * cooccurrence_score

        # Sorting
        itemcf_item_score_ranked = sorted(
            itemcf_item_score_dict.items(),
            key=lambda kv: kv[1], reverse=True
        )

        # Result
        num_selected_item = 0
        result = []

        for item, score in itemcf_item_score_ranked:
            num_selected_item += 1
            result.append(item)
            if num_selected_item == top_k:
                return result

        result_set = set(result)
        for item, freq in self.item_counter.most_common(top_k):
            if item not in result_set:
                num_selected_item += 1
                result.append(item)
            if num_selected_item == top_k:
                return result

    def predict(
        self,
        df,
        u2i_index_decay_fn, u2i_time_decay_fn,
        last_n, top_k, filtered=True
    ):
        result = {}
        for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
            user_id = getattr(row, self.user_id_colname)
            time = getattr(row, self.time_colname)
            result[user_id] = self.predict_each_user(
                user_id, time,
                u2i_index_decay_fn, u2i_time_decay_fn,
                last_n, top_k, filtered
            )
        return result


class SimpleTagRec:

    def __init__(
        self,
        user_to_tag,
        tag_to_item,
        default_item_list
    ):
        self.user_to_tag = user_to_tag
        self.tag_to_item = tag_to_item
        self.default_item_list = default_item_list

    def predict_each_user(
        self,
        user_id,
        user_tag_item_fn,
        top_k
    ):
        # Scoring
        item_score_dict = defaultdict(float)
        for tag, user_to_tag_weight in self.user_to_tag[user_id].items():
            for item, tag_to_item_weight in self.tag_to_item[tag].items():
                item_score_dict[item] = user_tag_item_fn(
                    user_to_tag_weight,
                    tag_to_item_weight
                )

        # Sorting
        item_score_ranked = sorted(
            item_score_dict.items(),
            key=lambda kv: kv[1], reverse=True
        )

        # Result
        num_selected_item = 0
        result = []

        for item, score in item_score_ranked:
            num_selected_item += 1
            result.append(item)
            if num_selected_item == top_k:
                return result

        result_set = set(result)
        for item in self.default_item_list:
            if item not in result_set:
                num_selected_item += 1
                result.append(item)
            if num_selected_item == top_k:
                return result

        return result

    def predict(
        self,
        user_id_list,
        user_tag_item_fn,
        top_k
    ):
        result = {}
        for user_id in tqdm(user_id_list):
            result[user_id] = self.predict_each_user(
                user_id,
                user_tag_item_fn,
                top_k
            )
        return result
