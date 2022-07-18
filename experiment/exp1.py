from collections import defaultdict
import math
import polars as pl
from src.candidate_generator import (
    ItemCF,
    Swing,
    SimpleTagRec
)
from src.utils import (
    exp_decay_fn,
    intersection,
    cg_eval
)


# Data
articles_df = pl.read_csv('./data/csv/articles.csv')
customers_df = pl.read_csv('./data/csv/customers.csv')
transactions_df = pl.read_csv('./data/csv/transactions_train.csv')

# Expressions
columns = ['t_dat', 'customer_id', 'article_id', 'price']
transactions_exprs = [
    pl.col('t_dat').str.strptime(pl.Date, '%Y-%m-%d').dt.epoch_days(),
    pl.col('article_id').cast(pl.Utf8).str.zfill(10)
]
articles_df_exprs = [
    pl.col('article_id').cast(pl.Utf8).str.zfill(10)
]

# Split
articles_df = articles_df.with_columns(articles_df_exprs)

train_df = (
    transactions_df
    .select(columns)
    .filter(pl.col('t_dat') <= '2019-06-20')
    .with_columns(transactions_exprs)
)

test_df = (
    transactions_df
    .select(columns)
    .filter(
        (pl.col('t_dat') > '2019-06-20') &
        (pl.col('t_dat') <= '2019-09-20')
    )
    .with_columns(transactions_exprs)
)

common_user_list = intersection(
    train_df['customer_id'].unique().to_list(),
    test_df['customer_id'].unique().to_list()
)

test_label_df = (
    test_df
    .filter(pl.col('customer_id').is_in(common_user_list))
    .select(['customer_id', 'article_id'])
    .groupby('customer_id')
    .agg_list()
)

labels = {
    row['customer_id']: row['article_id']
    for row in test_label_df.to_struct('rows')
}


# I2I
train_df_i2i = (
    train_df
    .filter(pl.col('t_dat') >= 18040)
    .sort('t_dat')
    .groupby('customer_id', maintain_order=True)
    .tail(50)
)

pred_df_i2i = pl.DataFrame({
    'customer_id': common_user_list,
    't_dat': [train_df_i2i['t_dat'].max()] * len(common_user_list)
})


# ItemCF
itemcf_model = ItemCF(
    user_id_colname='customer_id',
    item_id_colname='article_id',
    time_colname='t_dat'
)

itemcf_model.train(
    train_df_i2i.to_pandas(),
    positive_time_strength=1.00,
    negative_time_strength=0.90,
    i2i_index_decay_fn=lambda x: 1.0 ** (x - 1),
    i2i_time_decay_fn=lambda x: exp_decay_fn(1.0, 1.0, x / 365),
    user_frequency_regularization_fn=lambda x: 1.0 / math.log(10 + x),
    item_frequency_regularization_fn=lambda x, y: math.pow(x * y, 0.50)
)

itemcf_preds_1000 = itemcf_model.predict(
    pred_df_i2i.to_pandas(),
    u2i_index_decay_fn=lambda x: 1.0 ** (x - 1),
    u2i_time_decay_fn=lambda x: exp_decay_fn(1.0, 1.0, x / 365),
    last_n=20,
    top_k=1000,
    filtered=False
)

itemcf_preds_100 = {
    key: value[:100]
    for key, value in itemcf_preds_1000.items()
}

cg_eval(itemcf_preds_100, labels)
# 0.040688925056877795

cg_eval(itemcf_preds_1000, labels)
# 0.18072189867452992


# Swing
swing_model = Swing(
    user_id_colname='customer_id',
    item_id_colname='article_id',
    time_colname='t_dat'
)


swing_model.train(
    train_df_i2i.to_pandas(),
    positive_time_strength=1.00,
    negative_time_strength=0.90,
    i2i_index_decay_fn=lambda x: 1.0 ** (x - 1),
    i2i_time_decay_fn=lambda x: exp_decay_fn(1.0, 1.0, x / 365),
    user_intersection_threshold=(5, 20),
    item_intersection_threshold=(10, 100),
    user_intersection_weight_fn=lambda x: 1.0 / math.sqrt(10 + x),
    item_intersection_weight_fn=lambda x: 1.0 / (1 + x),
    user_frequency_regularization_fn=lambda x: 1.0 / math.log(10 + x),
    item_frequency_regularization_fn=lambda x: 1.0 / math.log(20 + x),
)


swing_preds_1000 = swing_model.predict(
    pred_df_i2i.to_pandas(),
    u2i_index_decay_fn=lambda x: 1.0 ** (x - 1),
    u2i_time_decay_fn=lambda x: exp_decay_fn(1.0, 1.0, x / 365),
    last_n=50,
    top_k=1000,
    filtered=False
)

swing_preds_100 = {
    key: value[:100]
    for key, value in swing_preds_1000.items()
}

cg_eval(swing_preds_100, labels)
# 0.038662561960376776

cg_eval(swing_preds_1000, labels)
# 0.19813483722579656


# Tag Based Recommendation

# User to Tag
user_to_tag_df = (
    train_df
    .filter(pl.col('customer_id').is_in(common_user_list))
    .filter(pl.col('t_dat') >= 18040)
    .join(articles_df.select(
        ['article_id', 'product_type_name']),
        on='article_id',
        how='left'
    )
    .groupby(['customer_id', 'product_type_name'])
    .agg([
        pl.count(),
        pl.avg('price')
    ])
)

user_to_tag_dict = defaultdict(dict)
for row in user_to_tag_df.to_struct('rows'):
    user_to_tag_dict[row['customer_id']][row['product_type_name']] = (
        row['count'], row['price']
    )

# Tag to Item
tag_to_item_df = (
    train_df
    .filter(pl.col('t_dat') >= 18040)
    .join(articles_df.select(
        ['article_id', 'product_type_name']),
        on='article_id',
        how='left'
    )
    .groupby(['product_type_name', 'article_id'])
    .agg([
        pl.count(),
        pl.avg('price')
    ])
    .filter(pl.col('count') >= 100)
)

tag_to_item_dict = defaultdict(dict)
for row in tag_to_item_df.to_struct('rows'):
    tag_to_item_dict[row['product_type_name']][row['article_id']] = (
        row['count'], row['price']
    )

# Default Recommendation
most_popular_item_list = (
    train_df
    .filter(pl.col('t_dat') >= 18040)
    .groupby('article_id')
    .count()
    .sort(by='count', reverse=True)
)['article_id'][:1000].to_list()


# SimpleTagRec Model
tag_model = SimpleTagRec(
    user_to_tag=user_to_tag_dict,
    tag_to_item=tag_to_item_dict,
    default_item_list=most_popular_item_list
)

tag_preds_1000 = tag_model.predict(
    common_user_list,
    user_tag_item_fn=lambda x, y: (
        math.log(1.0 + x[0]) * math.sqrt(y[0]) *
        min(max(x[1] / y[1], 0.5), 2.0)
    ),
    top_k=1000
)

tag_preds_100 = {
    key: value[:100]
    for key, value in tag_preds_1000.items()
}

cg_eval(tag_preds_100, labels)
# 0.03574208281467931

cg_eval(tag_preds_1000, labels)
# 0.19447772794197593
