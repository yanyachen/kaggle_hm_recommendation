import math
import yaml
import polars as pl
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_hub as hub
from src.layer import (
    StackingLayer,
    MultiTaskLoss,
    GravityRegularizationLayer,
    MultiTaskDSSM

)
from src.utils import (
    cg_eval,
    InnerProductSearcher
)


# Constant
FIRST_TIME_RUNNING = False
time_split = '2020-09-01'
user_feature_names = ['customer_id', 'age', 'postal_code']
item_specific_feature_names = [
    'article_id',
    'product_code', 'product_type_no',
    'graphical_appearance_no', 'colour_group_code',
    'perceived_colour_value_id', 'perceived_colour_master_id'
]
item_general_feature_names = [
    'product_group_name', 'department_no',
    'index_code', 'index_group_no', 'section_no', 'garment_group_no'
]
item_text_feature_names = [
    'detail_desc'
]
num_features = len(
    user_feature_names +
    item_specific_feature_names +
    item_general_feature_names +
    item_text_feature_names
)


if FIRST_TIME_RUNNING:
    # Initial Data Loading
    customers_df = pl.read_csv('./data/csv/customers.csv')
    articles_df = pl.read_csv('./data/csv/articles.csv')
    transactions_df = (
        pl
        .read_csv('./data/csv/transactions_train.csv')
        .filter(pl.col('t_dat') >= '2020-06-01')
    )

    # TODO Negative Sampling
    # TODO Feature Engineering

    # Data
    train_df = (
        transactions_df
        .filter(pl.col('t_dat') < time_split)
        .join(customers_df, on='customer_id', how='left')
        .join(articles_df, on='article_id', how='left')
        .select(
            user_feature_names +
            item_specific_feature_names +
            item_general_feature_names +
            item_text_feature_names
        )
    )
    train_df.write_csv('./temp/train.csv', has_header=True)

    item_df = (
        articles_df
        .filter(pl.col('article_id').is_in(transactions_df['article_id'].unique()))
        .select(
            item_specific_feature_names +
            item_general_feature_names +
            item_text_feature_names
        )
    )
    item_df.write_csv('./temp/item.csv', has_header=True)

    train_label_df = (
        transactions_df
        .filter(pl.col('t_dat') < time_split)
        .select(['customer_id', 'article_id'])
    )
    train_label_df.write_csv('./temp/train_label.csv', has_header=True)

    train_user_df = (
        customers_df
        .filter(pl.col('customer_id').is_in(train_label_df['customer_id'].unique()))
        .select(user_feature_names)
    )
    train_user_df.write_csv('./temp/train_user.csv', has_header=True)

    test_label_df = (
        transactions_df
        .filter(pl.col('t_dat') >= time_split)
        .select(['customer_id', 'article_id'])
    )
    test_label_df.write_csv('./temp/test_label.csv', has_header=True)

    test_user_df = (
        customers_df
        .filter(pl.col('customer_id').is_in(test_label_df['customer_id'].unique()))
        .select(user_feature_names)
    )
    test_user_df.write_csv('./temp/test_user.csv', has_header=True)

    # Statistics
    feature_cardinality_dict = {
        feature_name: train_df[feature_name].n_unique()
        for feature_name in (
            user_feature_names +
            item_specific_feature_names +
            item_general_feature_names
        )
    }

    with open('./temp/cardinality.yaml', 'w') as f:
        yaml.dump(feature_cardinality_dict, f)


# Data Loading
with open('./temp/cardinality.yaml', 'r') as f:
    feature_cardinality_dict = yaml.safe_load(f)

train_dataset = tf.data.experimental.make_csv_dataset(
    file_pattern='./temp/train.csv',
    batch_size=1024 // 4,
    column_defaults=[''] * num_features,
    num_epochs=1
)

item_df = pl.read_csv('./temp/item.csv')
test_user_df = pl.read_csv('./temp/train_user.csv')
test_label_df = (
    pl
    .read_csv('./temp/train_label.csv')
    .with_column(pl.col('article_id').cast(pl.Utf8))
    .groupby('customer_id')
    .agg_list()
)


# Feature Columns
def get_embedding_feature_columns(feature_names, feature_cardinality_dict):
    categorical_columns = [
        tf.feature_column.categorical_column_with_hash_bucket(
            key=feature_name,
            hash_bucket_size=int(feature_cardinality_dict[feature_name] * 2)
        )
        for feature_name in feature_names
    ]

    embedding_columns = [
        tf.feature_column.embedding_column(
            categorical_column=fc,
            dimension=int(math.log2(feature_cardinality_dict[fc.key])),
            combiner='sqrtn',
        )
        for fc in categorical_columns
    ]

    return embedding_columns


user_feature_embedding_columns = get_embedding_feature_columns(
    user_feature_names,
    feature_cardinality_dict
)

item_specific_feature_embedding_columns = get_embedding_feature_columns(
    item_specific_feature_names,
    feature_cardinality_dict
)

item_general_feature_embedding_columns = get_embedding_feature_columns(
    item_general_feature_names,
    feature_cardinality_dict
)

item_text_feature_embedding_columns = [
    hub.text_embedding_column_v2(
        key=feature_name,
        module_path='./temp/Wiki-words-250-with-normalization_2/',
        trainable=False
    )
    for feature_name in item_text_feature_names
]


# Embedding Layer
user_embedding_layer = tf.keras.layers.DenseFeatures(
    user_feature_embedding_columns,
    trainable=True
)

item_specific_embedding_layer = tf.keras.layers.DenseFeatures(
    item_specific_feature_embedding_columns,
    trainable=True
)

item_general_embedding_layer = tf.keras.layers.DenseFeatures(
    item_general_feature_embedding_columns,
    trainable=True
)

item_text_embedding_layer = tf.keras.layers.DenseFeatures(
    item_text_feature_embedding_columns,
    trainable=True
)


# Intermediate Model
user_embedding_model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Dense(64, activation=tf.keras.activations.gelu),
        tf.keras.layers.Dense(32, activation=tf.keras.activations.linear)
    ]
)

item_specific_embedding_model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Dense(64, activation=tf.keras.activations.gelu),
        tf.keras.layers.Dense(32, activation=tf.keras.activations.linear)
    ]
)

item_general_embedding_model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Dense(64, activation=tf.keras.activations.gelu),
        tf.keras.layers.Dense(32, activation=tf.keras.activations.linear)
    ]
)


# Embedding Model
user_model = StackingLayer(
    input_layers=[user_embedding_layer],
    output_layer=user_embedding_model
)
item_specific_model = StackingLayer(
    input_layers=[
        item_specific_embedding_layer,
        item_general_embedding_layer,
        item_text_embedding_layer,
    ],
    output_layer=item_specific_embedding_model
)
item_general_model = StackingLayer(
    input_layers=[item_general_embedding_layer],
    output_layer=item_general_embedding_model
)


# Multi Task Loss
model = MultiTaskDSSM(
    user_model,
    item_specific_model,
    item_general_model,
    specific_task_layer=tfrs.tasks.Retrieval(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
        batch_metrics=[tf.keras.metrics.AUC(name='auc_1', num_thresholds=10000, from_logits=True)],
        loss_metrics=[tf.keras.metrics.Mean(name='loss_1')],
        temperature=1.0,
        num_hard_negatives=None,
        remove_accidental_hits=False
    ),
    general_task_layer=tfrs.tasks.Retrieval(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
        batch_metrics=[tf.keras.metrics.AUC(name='auc_2', num_thresholds=10000, from_logits=True)],
        loss_metrics=[tf.keras.metrics.Mean(name='loss_2')],
        temperature=1.0,
        num_hard_negatives=None,
        remove_accidental_hits=False
    ),
    task_weighting_layer=MultiTaskLoss(),
    regularizaiton_layer=GravityRegularizationLayer(gamma=0.1)
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    jit_compile=False
)

model.fit(train_dataset)
# 17005/17005 [==============================] - 3446s 202ms/step - auc_1: 0.6486 - loss_1: 5.3990 - auc_2: 0.6299 - loss_2: 5.4261 - loss: 5.5866 - regularization_loss: 0.0000e+00 - total_loss: 5.5866
# 17005/17005 [==============================] - 3530s 207ms/step - auc_1: 0.6577 - loss_1: 5.4153 - auc_2: 0.6396 - loss_2: 5.4384 - loss: 5.6063 - regularization_loss: 0.0000e+00 - total_loss: 5.6063
# 4252/4252 [==============================] - 908s 212ms/step - auc_1: 0.6107 - loss_1: 6.8629 - auc_2: 0.5912 - loss_2: 6.8828 - loss: 7.1887 - regularization_loss: 0.0000e+00 - total_loss: 7.1887


# Evaluation
item_numpy_dict = {
    key: value.to_numpy().astype('str')
    for key, value in item_df.to_dict().items()
}
item_embedding = item_specific_model(item_numpy_dict)

test_user_numpy_dict = {
    key: value.to_numpy().astype('str')
    for key, value in test_user_df.to_dict().items()
}
test_user_embedding = user_model(test_user_numpy_dict)

test_label_dict = {
    row['customer_id']: row['article_id']
    for row in test_label_df.to_struct('rows')
}

searcher = InnerProductSearcher(
    item_embedding.numpy(),
    32,
    item_numpy_dict['article_id'].tolist()
)


test_user_pred = dict(zip(
    test_user_numpy_dict['customer_id'].tolist(),
    searcher.search(test_user_embedding.numpy(), 1000)
))


cg_eval(test_user_pred, test_label_dict)
# R@100: 0.008693028488801323
# R@1000: 0.07049066059866613
