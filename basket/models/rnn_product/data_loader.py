""" Custom DataLoader
"""

import sys
import traceback
import random
import multiprocessing
from collections import deque

import joblib
import numpy as np
from tqdm import tqdm

from ...preprocessing.prepare_users import USER_CHUNK
from ..utils import load_user_list

_MAIN = 0
_EVAL = 1
_TRAIN = 5
_TEST = 6
MAX_LENGTH = 10
N_THREAD = 4
DEBUG = 0


def prepare_user(user_id, sample_type, max_length=MAX_LENGTH):
    """ Create feature arrays for each user
    """
    user_data = joblib.load(
        "data/users/{}/{}.pkl".format(user_id // USER_CHUNK, user_id))
    max_order_number = user_data["max_order_number"]
    if sample_type == _EVAL:
        # Exclude the last order
        max_order_number -= 1
    days_since_prior_order = user_data["days_since_prior_order"] / 30.0
    time_features_f = []
    cate_features_f = []
    labels_f = []
    sequence_lengths = []
    cart_orders = user_data["cart_orders"][:, :max_order_number]
    cart_n_items = np.max(cart_orders, axis=0)
    # Prevent divide by zero
    cart_n_items[-1] = 1
    valid_products = 0
    for i, product in enumerate(user_data["product_list"]):
        if product["min_order_number"] == max_order_number:
            # Skip if the first order of the product is in the last global order
            continue
        # Assuming max_length = 10
        # Example 1: max_order_number = 15 & min(order_number) = 4
        #            => first_order_number =  5
        # Example 2: max_order_number = 4 & min(order_number) = 1
        #            => first_order_number =  1
        first_order_number = max(
            max_order_number - max_length, product["min_order_number"])
        sequence_length = max_order_number - first_order_number
        # time features: previously_ordered, (normalized)days_since_prior_order,
        #                relative_cart_order, previous_basket_size,
        #                previous_basket_reorer, hod, dow
        time_features_p = np.zeros((max_length, 1, 7)).astype(np.float32)
        labels_p = np.zeros((max_length, 1))
        cate_features_p = np.zeros((1, 4)).astype(np.int32)
        if product["aisle"] < 1:  # weird negative numbers
            product["aisle"] += 128 + 128
        # Reorders start from index 0
        # reorders = np.zeros(max_order_number + 1)
        # reorders[order_numbers] = 1
        orders = user_data["reordered"][i, :max_order_number]
        orders[product["min_order_number"] - 1] = 1
        # Cart orders also start from index 0
        relative_cart_orders = (
            cart_orders[i, :] / cart_n_items
        )
        relative_cart_orders[relative_cart_orders == 0] = 1
        labels_p[:sequence_length, 0] = (
            orders[first_order_number:max_order_number])
        # Ordered in the last time step
        time_features_p[:sequence_length, 0, 0] = (
            orders[(first_order_number - 1):(max_order_number - 1)])
        # days_since_prior_order starts with 0
        # but the index is the same because we start with the second order
        time_features_p[:sequence_length, 0, 1] = (
            days_since_prior_order[first_order_number:max_order_number]
        )
        time_features_p[:sequence_length, 0, 2] = (
            relative_cart_orders[(first_order_number - 1)
                                  :(max_order_number - 1)]
        )
        # basket_sizes start from 0
        time_features_p[:sequence_length, 0, 3] = np.clip(
            user_data["basket_sizes"][
                (first_order_number - 1):(max_order_number - 1)] / 50.0,
            0, 1
        )
        # basket_reorders start from 0
        time_features_p[:sequence_length, 0, 4] = np.clip(
            user_data["basket_reorders"][
                (first_order_number - 1):(max_order_number - 1)] / 50.0,
            0, 1
        )
        # hod starts with 0
        # but the index is the same because we start with the second order
        time_features_p[:sequence_length, 0, 5] = (
            user_data["hod"][first_order_number:max_order_number]
        )
        # dow starts with 0
        # but the index is the same because we start with the second order
        time_features_p[:sequence_length, 0, 6] = (
            user_data["dow"][first_order_number:max_order_number]
        )
        if np.sum(time_features_p[:, 0, 0]) == 0:
            # Skip if no recent purchases
            continue
        cate_features_p[0, 0] = user_id
        cate_features_p[0, 1] = product["product"]
        cate_features_p[0, 2] = product["department"]
        cate_features_p[0, 3] = product["aisle"]
        valid_products += 1
        time_features_f.append(time_features_p)
        cate_features_f.append(cate_features_p)
        labels_f.append(labels_p)
        sequence_lengths.append(int(sequence_length))
    # Add None as a product
    first_order_number = max(max_order_number - max_length, 1)
    sequence_length = max_order_number - first_order_number
    time_features_p = np.zeros((max_length, 1, 7)).astype(np.float32)
    labels_p = np.zeros((max_length, 1))
    cate_features_p = np.zeros((1, 4)).astype(np.int32)
    labels_p[:sequence_length, 0] = (
        1 * (user_data["basket_reorders"]
             [first_order_number:max_order_number] == 0)
    )
    time_features_p[:sequence_length, 0, 0] = (
        1 * (user_data["basket_reorders"]
             [(first_order_number - 1):(max_order_number - 1)] == 0)
    )
    time_features_p[:sequence_length, 0, 1] = (
        days_since_prior_order[first_order_number:max_order_number]
    )
    time_features_p[:sequence_length, 0, 2] = 1
    time_features_p[:sequence_length, 0, 3] = np.clip(
        user_data["basket_sizes"][(
            first_order_number - 1):(max_order_number - 1)] / 50.0,
        0, 1
    )
    time_features_p[:sequence_length, 0, 4] = np.clip(
        user_data["basket_reorders"][(
            first_order_number - 1):(max_order_number - 1)] / 100.0,
        0, 1
    )
    time_features_p[:sequence_length, 0, 5] = (
        user_data["hod"][first_order_number:max_order_number]
    )
    time_features_p[:sequence_length, 0, 6] = (
        user_data["dow"][first_order_number:max_order_number]
    )
    cate_features_p[0, 0] = user_id
    # Explicitly set to zeros
    cate_features_p[0, 1] = 0
    cate_features_p[0, 2] = 0
    cate_features_p[0, 3] = 0
    time_features_f.append(time_features_p)
    cate_features_f.append(cate_features_p)
    labels_f.append(labels_p)
    sequence_lengths.append(int(sequence_length))
    # print(sequence_lengths)
    return user_id, time_features_f, cate_features_f, labels_f, sequence_lengths


class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _worker_loop(user_queue, data_queue):
    while True:
        user_id, sample_type = user_queue.get()
        if user_id is None:
            data_queue.put(None)
            break
        try:
            res = prepare_user(user_id, sample_type)
        except Exception:
            data_queue.put((user_id, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put(res)


class InstacartDataLoader(object):
    """Iterates over the Instacart dataset once."""

    def __init__(self, *, train_users=None, eval_users=None, test_users=None,
                 shuffle=False, num_workers=0, batch_size=32, pbar=False):
        if test_users is not None and (train_users is not None or eval_users is not None):
            raise ValueError("Only one of (test_users) and (train_users, eval_users)"
                             "can be not None.")
        if train_users is not None or eval_users is not None:
            self.users = []
            if train_users is not None:
                self.users += [(user_id, _MAIN) for user_id in train_users]
            if eval_users is not None:
                self.users += [(user_id, _EVAL) for user_id in eval_users]
            self.mode = _TRAIN
        elif test_users is not None:
            self.users = [(user_id, _TEST) for user_id in test_users]
            self.mode = _TEST
        else:
            raise ValueError("At least on of test_users, train_user, and eval_users"
                             "need to be not None")
        if DEBUG:
            print("Number of users used:", len(self.users))
        if shuffle:
            random.shuffle(self.users)
        if pbar:
            self.pbar = tqdm(total=len(self.users))
        else:
            self.pbar = None
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.time_feature_cache = deque([])
        self.cate_feature_cache = deque([])
        self.label_cache = deque([])
        self.sequence_length_cache = deque([])
        self.send_idx = 0
        self.n_iter = 0
        if self.num_workers > 0:
            self.user_queue = multiprocessing.Queue()
            self.data_queue = multiprocessing.Queue()
            self.users_outstanding = 0
            self.shutdown = False
            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.user_queue, self.data_queue)
                ) for _ in range(self.num_workers)
            ]
            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()
            # prime the prefetch loop
            for _ in range(4 * self.num_workers):
                self._put_user()

    def _append_to_cache(self, user_res):
        _, time_features_u, cate_features_u, labels_u, sequence_lengths_u = user_res
        if not sequence_lengths_u:
            # Skip empty results
            return
        self.time_feature_cache += time_features_u
        self.cate_feature_cache += cate_features_u
        self.label_cache += labels_u
        self.sequence_length_cache += sequence_lengths_u

    def __next__(self):
        if self.num_workers == 0:
            while len(self.sequence_length_cache) < self.batch_size and self.send_idx < len(self.users):
                user_id, sample_type = self.users[self.send_idx]
                user_res = prepare_user(user_id, sample_type)
                self._append_to_cache(user_res)
                self.send_idx += 1
                if self.pbar:
                    self.pbar.update(1)
        else:
            while len(self.sequence_length_cache) < self.batch_size and self.users_outstanding > 0:
                user_res = self.data_queue.get()
                self.users_outstanding -= 1
                self._process_next_user(user_res)
        if self.sequence_length_cache:
            right_idx = min(len(self.sequence_length_cache), self.batch_size)
            batch = (
                np.concatenate(
                    [self.time_feature_cache.popleft() for _ in range(right_idx)], axis=1),
                np.concatenate(
                    [self.cate_feature_cache.popleft() for _ in range(right_idx)], axis=0)
            )
            labels = np.concatenate(
                [self.label_cache.popleft() for _ in range(right_idx)], axis=1)
            # Sort sequence_length in descending order
            sequence_length = np.array([
                self.sequence_length_cache.popleft() for _ in range(right_idx)])
            sorted_idx = np.argsort(sequence_length)[::-1]
            batch = (
                batch[0][:, sorted_idx, :],
                batch[1][sorted_idx, :]
            )
            self.n_iter += 1
            return batch, labels[:, sorted_idx], sequence_length[sorted_idx]
        else:
            if DEBUG:
                print("Total batches", self.n_iter)
            if self.num_workers > 0:
                self._shutdown_workers()
            if self.pbar:
                self.pbar.close()
            raise StopIteration

    def __iter__(self):
        return self

    def _put_user(self):
        assert self.users_outstanding < 4 * self.num_workers
        if self.send_idx == len(self.users):
            return
        self.user_queue.put(self.users[self.send_idx])
        self.users_outstanding += 1
        self.send_idx += 1
        if self.pbar:
            self.pbar.update(1)

    def _process_next_user(self, user_res):
        self._put_user()
        if isinstance(user_res[1], ExceptionWrapper):
            raise user_res[1].exc_type(user_res[1].exc_msg)
        self._append_to_cache(user_res)

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            for _ in self.workers:
                self.user_queue.put((None, None))

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()
        if self.pbar:
            self.pbar.close()


def validate_dataset():
    """ Simple test function to the data loader
    """
    train_users, test_users = load_user_list(train_sample_ratio=1.)
    train_loader = InstacartDataLoader(
        train_users=train_users,
        eval_users=[], shuffle=False,
        batch_size=512, num_workers=N_THREAD, pbar=True)
    for batch, labels, sequence_lengths in train_loader:
        assert len(batch) == 3
        assert labels.shape[0] == MAX_LENGTH
        assert len(sequence_lengths) == labels.shape[1]
        assert all(sequence_lengths > 0)
        assert all(sequence_lengths <= MAX_LENGTH)
        assert np.array_equal(
            np.sort(sequence_lengths)[::-1],
            sequence_lengths
        )
        for i in range(batch[0].shape[1]):
            assert np.array_equal(
                batch[0][1:sequence_lengths[i], i, 0],
                labels[:(sequence_lengths[i] - 1), i]
            )
        assert np.all(batch[0][:, :, 0] <= 1)   # bought in the last order
        assert np.all(batch[0][:, :, 1] <= 1)   # days since the last order
        assert np.all(batch[0][:, :, 2] <= 1)   # relative cart position
        assert np.all(batch[0][:, :, 3] <= 1)   # basket size
        assert np.all(batch[0][:, :, 4] <= 1)   # basket reorder
        assert np.all(batch[0][:, :, 5] < 24)   # hod
        assert np.all(batch[0][:, :, 6] < 7)    # dow
        assert np.all(batch[1][:, 0] < 206210)  # user_id
        assert np.all(batch[1][:, 1] < 49689)   # product_id
        assert np.all(batch[1][:, 2] < 23)      # department_id
        assert np.all(batch[1][:, 3] < 135)     # aisle_id
    eval_loader = InstacartDataLoader(
        train_users=[],
        eval_users=test_users + train_users, shuffle=False,
        batch_size=512, num_workers=N_THREAD, pbar=True)
    for batch, labels, sequence_lengths in eval_loader:
        # pick a random row to validate
        sample_idx = random.randint(0, labels.shape[0])
        user_id = batch[1][sample_idx, 0]
        user_data = joblib.load(
            "cache/users/{}/{}.pkl".format(user_id // USER_CHUNK, user_id))
        min_order_number = 1
        for prod in user_data["product_list"]:
            if prod["product"] == batch[1][sample_idx, 1]:
                min_order_number = prod["min_order_number"]
        assert (
            sequence_lengths[sample_idx] == min(
                MAX_LENGTH, user_data["max_order_number"] - 1 - min_order_number)
        )


if __name__ == "__main__":
    validate_dataset()
