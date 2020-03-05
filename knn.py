import numpy as np
from sklearn.neighbors import KDTree


class KNN:
    def __init__(self, n_neighbors=5, leaf_size=30, p=2, algorithm="kd_tree"):
        self._n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.p = p
        self.algorithm = algorithm
        self.tree = None
        self.X_train = None
        self.y_train = None

    @staticmethod
    def minkowski_dist(vec1, vec2, p):
        return np.sum(np.abs(vec1 - vec2) ** p) ** (1.0 / p)

    @staticmethod
    def mode(array, order_priority=False):
        if order_priority:
            u, uidx = np.unique(array, return_index=True)
            u_unsorted = array[np.sort(uidx)]  # unique values in order of occurrence

            # replace values in the array with their order of occurrence
            d = dict(zip(u_unsorted, np.arange(len(u_unsorted))))
            arr_renamed = np.array([d[x] for x in array])

            mode = u_unsorted[np.argmax(np.bincount(arr_renamed))]

        else:
            mode = np.argmax(np.bincount(array))

        return mode

    """
    :param X - np.array of shape (n_objects, n_features)
    :param y - np.array of shape (n_objects, 1) or (n_objects,)
    """
    def fit(self, X, y):
        self.y_train = y.reshape(-1)

        if self.algorithm == "kd_tree":
            self.tree = KDTree(X, leaf_size=self.leaf_size, metric="minkowski", p=self.p)

        elif self.algorithm == "brute":
            self.X_train = X

    """
    :param X - np.array of shape (n_objects, n_features)
    :param k (int, optional) - number of neighbors. Default by initialization.
    :param exclude_nearest (bool, optional) - do not count the vote of the nearest neighbor (False by default)
    :param order_priority (bool, optional) - in case of multiple candidates with equal votes give priority 
                                             to the first occurred value. Otherwise, smaller index is given priority.
                            Example k=4: [9,9,2,2,3] order_priority=False => prediction = 2
                                                     order_priority=True (by default) => prediction = 9
                            Useful if neighbors are sorted by distance in ascending order.
    :return predictions - np.array of shape (n_objects,)
    """
    def predict(self, X, k=None, exclude_nearest=False, order_priority=True):
        if not k:
            k = self._n_neighbors

        _, inds = self.get_k_neighbors(X, k=k)
        neighbors_y = self.y_train[inds]

        if exclude_nearest:
            neighbors_y = neighbors_y[:, 1:]

        predictions = np.apply_along_axis(lambda x: self.mode(x, order_priority), axis=1, arr=neighbors_y)

        return predictions

    """
    :param X - np.array of shape (n_objects, n_features)
    :param y - np.array of shape (n_objects, 1) or (n_objects,)
    :return predictions - np.array of shape (n_objects,)
    """
    def fit_predict(self, X, y, k=None, exclude_nearest=False, order_priority=True):
        if not k:
            k = self._n_neighbors

        self.fit(X, y)

        return self.predict(X, k, exclude_nearest=exclude_nearest, order_priority=order_priority)

    """
    :param X - np.array of shape (n_queries, n_features)
    :return dist - distances. np.array of shape (n_queries, n_neighbors)
    :return ind - indices. np.array of shape (n_queries, n_neighbors)
    """
    def get_k_neighbors(self, X, k):
        if self.algorithm == "kd_tree":
            dists_arr = []
            inds_arr = []

            for idx, row in enumerate(X):
                add = 1  # 1 extra neighbor to check if it has the same dist as the last one
                dists, inds = self.tree.query(row.reshape(1, -1), k=k+add)

                while np.allclose(dists[0][-1], dists[0][-2]):
                    add += 1
                    dists, inds = self.tree.query(row.reshape(1, -1), k=k+add)

                # now we have k+i >= k+1 elements and the last 2 differ, so we can cut the last one
                dists = dists[0, :-1]  # 0 as we consider one row
                inds = inds[0, :-1]
                add -= 1

                # if we still have an extra element, then we have a tie
                if add > 0:
                    # find out how many of the ending elements are tied
                    num_tied = 1
                    i = len(dists)-1
                    while np.allclose(dists[i], dists[i-1]) and i > 1:
                        num_tied += 1
                        i -= 1

                    # We need to delete [add] number of elements out of [num_tied] last elements.
                    # Approach: consider sub-array of tied elements and apply Partition() alg to put indexes of
                    # [add] largest elements to the right of the (num_tied - add)-th element.
                    partitioned_tied_subarray = np.argpartition(inds[-num_tied:], num_tied - add)
                    # select indices of the elements to be deleted and adjust for the whole array length
                    delete_inds = partitioned_tied_subarray[-add:] + (len(inds) - num_tied)

                    inds = np.delete(inds, delete_inds)
                    dists = np.delete(dists, delete_inds)

                inds_arr.append(inds)
                dists_arr.append(dists)

            return np.array(dists_arr), np.array(inds_arr)

        elif self.algorithm == "brute":
            dists_arr = []
            inds_arr = []

            for idx, x in enumerate(X):
                best_dists = np.array([self.minkowski_dist(x, self.X_train[j], p=self.p) for j in range(k)])
                indices = np.array(range(k))

                for i in range(k, len(self.X_train)):
                    distance = self.minkowski_dist(x, self.X_train[i], p=self.p)

                    if distance < np.max(best_dists):
                        index_to_replace = np.argmax(best_dists)
                        best_dists[index_to_replace] = distance
                        indices[index_to_replace] = i

                # sort by distances in ascending order, both indices and dists
                indices = indices[np.argsort(best_dists)]
                best_dists = np.sort(best_dists)

                dists_arr.append(best_dists)
                inds_arr.append(indices)

            return np.array(dists_arr), np.array(inds_arr)
