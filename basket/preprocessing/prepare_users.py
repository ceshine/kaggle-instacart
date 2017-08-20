"""Write a pickle file storing order information for each user.
"""

import os

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

IDIR = 'data/raw/'
USER_CHUNK = 1000
DEBUG = 0
N_THREAD = 4


def read_data():
    print('loading prior')
    priors = pd.read_csv(
        IDIR + 'order_products__prior.csv',
        dtype={
            'order_id': np.int32,
            'product_id': np.int32,
            'reordered': np.int8,
            'add_to_cart_order': np.int16,
            'user_id': np.int32
        }
    )
    print('loading orders')
    orders = pd.read_csv(
        IDIR + 'orders.csv',
        dtype={
            'order_id': np.int32,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'order_number': np.int8,
            'days_since_prior_order': np.float32,
            'eval_set': 'category',
            'user_id': np.int32
        })
    print('loading products')
    products = pd.read_csv(
        IDIR + 'products.csv',
        usecols=("product_id", "aisle_id", "department_id"),
        dtype={
            'aisle_id': np.int8,
            'department_id': np.int8,
            'product_id': np.int32,
        }
    )
    print('priors {}x{}: {}'.format(len(priors), len(
        priors.columns), ', '.join(priors.columns)))
    print('orders {}x{}: {}'.format(len(orders), len(
        orders.columns), ', '.join(orders.columns)))
    priors = pd.merge(
        priors,
        orders,
        how="left",
        on="order_id"
    ).drop("order_id", axis=1)
    # Combine prior orders with product info
    priors = pd.merge(priors, products, how="left", on="product_id")
    if DEBUG:
        assert np.sum(priors.eval_set.isnull()) == 0
        assert np.sum(priors.department_id.isnull()) == 0
    priors["days_since_prior_order"] = priors.loc[
        :, "days_since_prior_order"].fillna(0).astype(np.int8)
    priors = priors.sort_values(["user_id", "order_number"])
    print('loading train')
    train = pd.read_csv(
        IDIR + 'order_products__train.csv',
        dtype={
            'order_id': np.int32,
            'product_id': np.int32,
            'reordered': np.int8,
            'add_to_cart_order': np.int16
        }
    )
    print('train {}x{}: {}'.format(len(train), len(
        train.columns), ', '.join(train.columns)))
    # To get corresponding user_ids for each order
    train = train.merge(
        orders[["order_id", "user_id"]], on="order_id")
    # This is the training labels
    train_labels = train[["user_id", "product_id", "reordered"]].sort_values(
        ["user_id", "product_id"]).groupby("user_id").apply(
        lambda df: set(df[df.reordered == 1]["product_id"].tolist())
            if df["reordered"].sum() > 0 else set(["None"])
    ).reset_index().rename(columns={0: "truth"})
    # The last order of every users
    orders = orders[orders.eval_set != "prior"]
    if DEBUG:
        assert not any(orders.days_since_prior_order.isnull())
    orders["days_since_prior_order"] = orders["days_since_prior_order"].astype(
        np.int8)
    # Combine training labels and the last order
    orders = orders.merge(train_labels, on="user_id", how="left")
    orders = orders.set_index("user_id")
    return priors, orders


def process_user(df, order):
    max_order_number = order["order_number"]
    if DEBUG:
        assert df["order_number"].nunique() + 1 == max_order_number
    order_info = df[["order_number", "days_since_prior_order",
                     "order_dow", "order_hour_of_day"]].drop_duplicates()
    # Append last order information
    days_since_prior_order = np.concatenate(
        [order_info["days_since_prior_order"].values,
            [order["days_since_prior_order"]]], axis=0
    )
    dow = np.concatenate(
        [order_info["order_dow"].values, [order["order_dow"]]], axis=0
    )
    hod = np.concatenate(
        [order_info["order_hour_of_day"].values,
            [order["order_hour_of_day"]]], axis=0
    )
    product_list = df.groupby(["product_id"]).apply(
        process_user_product,
        max_order_number=max_order_number
    ).tolist()
    # Combine product series into a 2-d array
    reordered = np.stack(
        [x["reordered"] for x in product_list], axis=0
    )
    cart_orders = np.stack(
        [x["cart_orders"] for x in product_list], axis=0
    )
    basket_reorders = np.sum(reordered, axis=0)
    basket_sizes = np.copy(basket_reorders)
    # Fill last element only if the user's in the training set
    if isinstance(order["truth"], set):
        basket_sizes[-1] = len(order["truth"])
        basket_reorders[-1] = len(
            order["truth"].intersection(
                set([prod["product"] for prod in product_list])
            )
        )
    for i, prod_dict in enumerate(product_list):
        del prod_dict["reordered"]
        del prod_dict["cart_orders"]
        # Take the first order of each product into account
        basket_sizes[prod_dict["min_order_number"] - 1] += 1
        # Fill in last order labels
        if isinstance(order["truth"], set):
            reordered[i, -1] = 1 * \
                (prod_dict["product"] in order["truth"])
    res = dict(
        max_order_number=max_order_number,
        days_since_prior_order=days_since_prior_order,
        dow=dow, hod=hod,
        cart_orders=cart_orders,
        reordered=reordered,
        product_list=product_list,
        basket_sizes=basket_sizes,
        basket_reorders=basket_reorders
    )
    # Write the pickle file
    joblib.dump(
        res, "data/users/{}/{}.pkl".format(
            df.user_id.iloc[0] // USER_CHUNK, df.user_id.iloc[0])
    )


def process_user_product(df, max_order_number):
    if DEBUG:
        if df.shape[0] > 1:
            assert np.all(df.reordered[1:] == 1)
    cart_orders = np.zeros(max_order_number)
    reordered = np.zeros(max_order_number)
    cart_orders[df.order_number.values - 1] = df.add_to_cart_order.values
    if df.shape[0] > 1:
        reordered[df.order_number.values[1:] - 1] = 1
    min_order_number = df.order_number.min()
    aisle = df.aisle_id.iloc[0]
    department = df.department_id.iloc[0]
    return dict(
        reordered=reordered,
        product=df.product_id.iloc[0],
        cart_orders=cart_orders,
        aisle=aisle,
        department=department,
        min_order_number=min_order_number
    )


def main():
    priors, orders = read_data()

    priors = priors.set_index("user_id", drop=False)
    for i in range(priors["user_id"].max() // USER_CHUNK + 1):
        os.makedirs("data/users/{}/".format(i), exist_ok=True)
    joblib.Parallel(n_jobs=N_THREAD)(
        joblib.delayed(process_user)(
            priors.loc[user_id], orders.loc[user_id]
        ) for user_id in tqdm(set(orders.index))
    )


if __name__ == "__main__":
    main()
