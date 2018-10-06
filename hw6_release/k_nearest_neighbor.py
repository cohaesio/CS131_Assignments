import numpy as np

from scipy.spatial.distance import cdist
def compute_distances(X1, X2):
    """Compute the L2 distance between each point in X1 and each point in X2.
    It's possible to vectorize the computation entirely (i.e. not use any loop).

    Args:
        X1: numpy array of shape (M, D) normalized along axis=1
        X2: numpy array of shape (N, D) normalized along axis=1

    Returns:
        dists: numpy array of shape (M, N) containing the L2 distances.
    """
    M = X1.shape[0]
    N = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]

    dists = np.zeros((M, N))

    # YOUR CODE HERE
    # Compute the L2 distance between all X1 features and X2 features.
    # Don't use any for loop, and store the result in dists.
    #
    # You should implement this function using only basic array operations;
    # in particular you should not use functions from scipy.
    #
    # HINT: Try to formulate the l2 distance using matrix multiplication

    # 引入scipy的cdist函数进行两个输入的距离计算
    dists = cdist(X1, X2,metric='euclidean')

    # mikucy方式，运行速度较快
    # X1_square = np.sum(np.square(X1), axis=1)
    # X2_square = np.sum(np.square(X2), axis=1)
    # dists = np.sqrt(X1_square.reshape(-1, 1) - 2 * X1.dot(X2.T) + X2_square)
    # END YOUR CODE

    assert dists.shape == (M, N), "dists should have shape (M, N), got %s" % dists.shape


    return dists


from collections import Counter
def predict_labels(dists, y_train, k=1):
    """Given a matrix of distances `dists` between test points and training points,
    predict a label for each test point based on the `k` nearest neighbors.

    Args:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j] gives
               the distance betwen the ith test point and the jth training point.

    Returns:
        y_pred: A numpy array of shape (num_test,) containing predicted labels for the
                test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test, num_train = dists.shape
    y_pred = np.zeros(num_test, dtype=np.int)

    for i in range(num_test):
        # A list of length k storing the labels of the k nearest neighbors to
        # the ith test point.
        closest_y = []
        # Use the distance atrix to find the k mnearest neighbors of the ith
        # testing point, and use self.y_train to find the labels of these
        # neighbors. Store these labels in closest_y.
        # Hint: Look up the function numpy.argsort.

        # Now that you have found the labels of the k nearest neighbors, you
        # need to find the most common label in the list closest_y of labels.
        # Store this label in y_pred[i]. Break ties by choosing the smaller
        # label.

        # YOUR CODE HERE

        # # argsort能返回数组值从小到大的索引值,构建列表
        # closest_y = np.argsort(dists[i,:])[:k].tolist()
        #
        # # 判断前k个元素中频次最多的元素
        # # 调用Counter函数
        #
        # # 方法一
        # # a = np.array([1, 9, 3, 9, 2, 9, 1, 1, 9, 9, 2, 1])
        # # counts = np.bincount(a)
        # # print(np.argmax(counts))
        #
        # # 方法二
        # from collections import Counter
        # # a = [1, 2, 3, 1, 2, 1, 1, 1, 3, 2, 2, 1]
        # # b = Counter(a)
        # # print
        # # b.most_common(1)
        #
        # y_pred[i] = y_train[np.bincount(closest_y).argmax()]

        # mikucy做法
        idx = np.argsort(dists[i, :])
        for j in range(k):
            closest_y.append(y_train[idx[j]])
        y_pred[i] = max(set(closest_y), key=closest_y.count)

        # END YOUR CODE

    return y_pred


def split_folds(X_train, y_train, num_folds):
    """Split up the training data into `num_folds` folds.

    The goal of the functions is to return training sets (features and labels) along with
    corresponding validation sets. In each fold, the validation set will represent (1/num_folds)
    of the data while the training set represent (num_folds-1)/num_folds.
    If num_folds=5, this corresponds to a 80% / 20% split.

    For instance, if X_train = [0, 1, 2, 3, 4, 5], and we want three folds, the output will be:
        X_trains = [[2, 3, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 2, 3]]
        X_vals = [[0, 1],
                  [2, 3],
                  [4, 5]]

    Args:
        X_train: numpy array of shape (N, D) containing N examples with D features each
        y_train: numpy array of shape (N,) containing the label of each example
        num_folds: number of folds to split the data into

    returns:
        X_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds, D)
        y_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds)
        X_vals: numpy array of shape (num_folds, train_size / num_folds, D)
        y_vals: numpy array of shape (num_folds, train_size / num_folds)

    """
    assert X_train.shape[0] == y_train.shape[0]

    validation_size = X_train.shape[0] // num_folds
    training_size = X_train.shape[0] - validation_size

    X_trains = np.zeros((num_folds, training_size, X_train.shape[1]))
    y_trains = np.zeros((num_folds, training_size), dtype=np.int)
    X_vals = np.zeros((num_folds, validation_size, X_train.shape[1]))
    y_vals = np.zeros((num_folds, validation_size), dtype=np.int)

    # YOUR CODE HERE
    # Hint: You can use the numpy array_split function.
    # 用array_split 函数分割为n个片段，然后分序进行拼接
    X_list = np.array_split(X_train,num_folds,axis=0)
    y_train = y_train.reshape(-1, 1)
    y_list = np.array_split(y_train, num_folds, axis=0)
    for i in range(num_folds):
        X_trains[i,:,:] = np.vstack(X_list[:i] + X_list[i+1:])
        X_vals[i,:,:] = X_list[i]

        y_trains[i] = np.vstack(y_list[:i] + y_list[i+1:])[:,0]
        y_vals[i] = y_list[i][:,0]

    # END YOUR CODE

    return X_trains, y_trains, X_vals, y_vals
