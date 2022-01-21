import math
import heapq  # for retrieval topK
import multiprocessing
from functools import partial
import numpy as np


class MFModel:
    """A matrix factorization model trained using SGD and negative sampling."""

    def __init__(self, num_user, num_item, embedding_dim, reg, stddev):
        """Initializes MFModel.

        Args:
          num_user: the total number of users.
          num_item: the total number of items.
          embedding_dim: the embedding dimension.
          reg: the regularization coefficient.
          stddev: embeddings are initialized from a random distribution with this
            standard deviation.
        """
        self.user_embedding = np.random.normal(
            0, stddev, (num_user, embedding_dim))
        self.item_embedding = np.random.normal(
            0, stddev, (num_item, embedding_dim))
        self.user_bias = np.zeros([num_user])
        self.item_bias = np.zeros([num_item])
        self.bias = 0.0
        self.reg = reg

    def _predict_one(self, user, item):
        """Predicts the score of a user for an item."""
        return (self.bias + self.user_bias[user] + self.item_bias[item] +
                np.dot(self.user_embedding[user], self.item_embedding[item]))

    def predict(self, pairs):
        """Computes predictions for a given set of user-item pairs.

        Args:
          pairs: A pair of lists (users, items) of the same length.

        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        for i in range(num_examples):
            predictions[i] = self._predict_one(pairs[0][i], pairs[1][i])
        return predictions

    def fit(self, positive_pairs, learning_rate, num_negatives):
        """Trains the model for one epoch.

        Args:
          positive_pairs: an array of shape [n, 2], each row representing a positive
            user-item pair.
          learning_rate: the learning rate to use.
          num_negatives: the number of negative items to sample for each positive.

        Returns:
          The logistic loss averaged across examples.
        """
        # Convert to implicit format and sample negatives.
        user_item_label_matrix = self._convert_ratings_to_implicit_data(
            positive_pairs, num_negatives)
        np.random.shuffle(user_item_label_matrix)

        # Iterate over all examples and perform one SGD step.
        num_examples = user_item_label_matrix.shape[0]
        reg = self.reg
        lr = learning_rate
        sum_of_loss = 0.0
        for i in range(num_examples):
            (user, item, rating) = user_item_label_matrix[i, :]
            user_emb = self.user_embedding[user]
            item_emb = self.item_embedding[item]
            prediction = self._predict_one(user, item)

            if prediction > 0:
                one_plus_exp_minus_pred = 1.0 + np.exp(-prediction)
                sigmoid = 1.0 / one_plus_exp_minus_pred
                this_loss = (np.log(one_plus_exp_minus_pred) +
                             (1.0 - rating) * prediction)
            else:
                exp_pred = np.exp(prediction)
                sigmoid = exp_pred / (1.0 + exp_pred)
                this_loss = -rating * prediction + np.log(1.0 + exp_pred)

            grad = rating - sigmoid

            self.user_embedding[user, :] += lr * \
                (grad * item_emb - reg * user_emb)
            self.item_embedding[item, :] += lr * \
                (grad * user_emb - reg * item_emb)
            self.user_bias[user] += lr * (grad - reg * self.user_bias[user])
            self.item_bias[item] += lr * (grad - reg * self.item_bias[item])
            self.bias += lr * (grad - reg * self.bias)

            sum_of_loss += this_loss

        # Return the mean logistic loss.
        return sum_of_loss / num_examples

    def _convert_ratings_to_implicit_data(self, positive_pairs, num_negatives):
        """Converts a list of positive pairs into a two class dataset.

        Args:
          positive_pairs: an array of shape [n, 2], each row representing a positive
            user-item pair.
          num_negatives: the number of negative items to sample for each positive.
        Returns:
          An array of shape [n*(1 + num_negatives), 3], where each row is a tuple
          (user, item, label). The examples are obtained as follows:
          To each (user, item) pair in positive_pairs correspond:
          * one positive example (user, item, 1)
          * num_negatives negative examples (user, item', 0) where item' is sampled
            uniformly at random.
        """
        num_items = self.item_embedding.shape[0]
        num_pos_examples = positive_pairs.shape[0]
        training_matrix = np.empty([num_pos_examples * (1 + num_negatives), 3],
                                   dtype=np.int32)
        index = 0
        for pos_index in range(num_pos_examples):
            u = positive_pairs[pos_index, 0]
            i = positive_pairs[pos_index, 1]

            # Treat the rating as a positive training instance
            training_matrix[index] = [u, i, 1]
            index += 1

            # Add N negatives by sampling random items.
            # This code does not enforce that the sampled negatives are not present in
            # the training data. It is possible that the sampling procedure adds a
            # negative that is already in the set of positives. It is also possible
            # that an item is sampled twice. Both cases should be fine.
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                training_matrix[index] = [u, j, 0]
                index += 1
        return training_matrix

    def evaluate(self, test_ratings, test_negatives, K, num_thread=1):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(partial(self._eval_one_rating, test_ratings=test_ratings, test_negatives=test_negatives, K=K), range(len(test_ratings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return np.array(hits).mean(), np.array(ndcgs).mean()

    def _eval_one_rating(self, idx, test_ratings, test_negatives, K):
        rating = test_ratings[idx]
        items = test_negatives[idx]
        u = rating[0]
        true_item = rating[1]
        items.append(true_item)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u, dtype='int32')
        predictions = self.predict([users, np.array(items)])
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.hitrate(ranklist, true_item)
        ndcg = self.ndcg(ranklist, true_item)
        return (hr, ndcg)

    @staticmethod
    def hitrate(ranklist, true_item):
        """recall"""
        for item in ranklist:
            if item == true_item:
                return 1
        return 0

    @staticmethod
    def ndcg(ranklist, true_item):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == true_item:
                return math.log(2) / math.log(i+2)
        return 0