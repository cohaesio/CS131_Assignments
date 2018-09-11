import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    num = 0
    for n in range(num_iters):
        ### YOUR CODE HERE

        # 分配每个点至最近的聚类中心
        for i in range(N):
            dist = np.linalg.norm(features[i]-centers, axis=1)
            assignments[i] = np.argmin(dist)

        # 重新计算每个聚类中心的位置
        newCenters = np.zeros((k, D))
        for idx in range(k):
            # 每个聚类的所有下标
            index = np.where(assignments==idx)
            # 计算均值
            newCenters[idx] = np.mean(features[index],axis=0)

        # 如果不再移动则停止迭代
        # 还可以用allclose函数进行判断
        if np.linalg.norm(newCenters-centers) < 1e-20:
            break
        else:
            centers = newCenters

        num = n
        ### END YOUR CODE
    # 输出迭代次数
    print("迭代次数",n)
    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE

        # 计算每个点最近的
        # 向量化的思路，把N个点（特征）放入一个矩阵中进行计算
        featureMap = np.tile(features, (k,1))
        # print(features.shape)
        centerMap = np.repeat(centers, N, axis=0)
        # print((centerMap.shape))
        dist = np.linalg.norm(featureMap-centerMap, axis=1).reshape(k,N)
        assignments = np.argmin(dist, axis=0)
        # print(dist.shape)

        # 重新计算每个聚类中心的位置
        newCenters = np.zeros((k, D))
        for idx in range(k):
            # 每个聚类的所有下标
            index = np.where(assignments == idx)
            # 计算均值
            newCenters[idx] = np.mean(features[index], axis=0)

        # 如果不再移动则停止迭代
        # 还可以用allclose函数进行判断
        # if np.linalg.norm(newCenters - centers) < 1e-20:
        if np.allclose(newCenters, centers):
            break
        else:
            centers = newCenters


        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to defeine distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        ### YOUR CODE HERE

        # 构建邻近度矩阵
        # print(centers.shape)
        dist = pdist(centers) # 欧氏距离
        # print(dist.shape)
        distMatrix = squareform(dist)
        distMatrix = np.where(distMatrix!=0.0, distMatrix, 1e5) # 因为squareform构建的邻近度矩阵上对角线是零，影响了最短距离的判断，此处做处理
        # print(distMatrix.shape)

        # 获取最小值所在的行和列，即距离最近的两个簇的index，取较小的那个作为保留，较大的那个进行合并
        minRow,minCol = np.unravel_index(distMatrix.argmin(), distMatrix.shape)
        # print(minRow,minCol)
        saveIdx = min(minRow,minCol)
        mergeIdx = max(minRow,minCol)
        # 将簇mergeIdx的点分配给saveIdx所在簇
        assignments = np.where(assignments != mergeIdx, assignments, saveIdx)
        # 因为要删除一个簇mergeIdx，所以下标的变化为:小于mergeIdx的不改变，大于saveIdx的需要减一
        assignments = np.where(assignments < mergeIdx, assignments, assignments - 1)

        # 删除被合并的簇所在中心
        centers = np.delete(centers,mergeIdx, axis=0)

        # 重新计算新的簇的中心
        # 题目提示中提到了，考虑到运行效率，只需要对进行归并后的簇计算新的中心，别的簇并没有进行操作，所以不影响中心的变化
        saveIdxIndecies = np.where(assignments == saveIdx)
        centers[saveIdx] = np.mean(features[saveIdxIndecies],axis=0)

        # 别忘了迭代终止条件
        n_clusters -= 1


        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    # 图像像素已经转换为浮点数
    # 只需要将图像转换为特征序列即可
    features = np.reshape(img, (H*W, C))
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    # 构建二维的点集，三维是meshgrid
    # positonMap =
    # 构建成一维的特征序列
    position = np.dstack(np.mgrid[0:H, 0:W]).reshape((H*W,2))

    # 拼接特征序列
    features[:,0:C] = np.reshape(color, (H*W, C))
    features[:,C:C+2] = position

    # 每个维度进行归一化
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0))

    ### END YOUR CODE

    return features

from skimage import color
def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE

    # 颜色特征
    H, W, C = img.shape
    colors = img_as_float(img)

    # 位置特征
    # 构建二维的点集，三维是meshgrid
    # positonMap =
    # 构建成一维的特征序列
    position = np.dstack(np.mgrid[0:H, 0:W]).reshape((H*W,2))


    # 梯度特征（梯度相加形式，也可以用模值）
    grayImg = color.rgb2gray(img)

    gradient = np.gradient(grayImg)
    gradient = np.abs(gradient[0]) + np.abs(gradient[1])
    # gradient = np.reshape(gradient,(H*W,-1))
    # print(gradient.shape)
    # print(img.shape)
    # 拼接特征序列
    features = np.zeros((H * W, C + 3))

    features[:,0:C] = np.reshape(colors, (H*W, C))
    features[:,C:C+2] = position
    features[:, C+2] = gradient.reshape((H*W))

    # 每个维度进行归一化
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0))

    ### END YOUR CODE
    return features
    

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    accuracy = np.mean(mask == mask_gt)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
