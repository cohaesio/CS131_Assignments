import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    # faster conv in hw1
    # 卷积向量化
    kernel_Vec = kernel.reshape(Hk*Wk,1)
    # 计算块向量化，并组成大矩阵
    vec_Mat = np.zeros((Hi*Wi, Hk*Wk))
    for row in range(Hi):
        for col in range(Wi):
            vec_Mat[row*Wi + col, :] = padded[row:row+Hk,col:col+Wk].reshape(1, Hk*Wk)
    # 进行内积计算
    result = np.dot(vec_Mat, kernel_Vec)
    # reshape恢复图像尺寸
    out = result.reshape(Hi,Wi)

    # 注意卷积形式不同内核也需要注意
    # kernel = np.flip(kernel, 0)
    # kernel = np.flip(kernel, 1)
    #
    # for Y in range(Hi):
    #     for X in range(Wi):
    #         # trans X,Y to indexs in imgBig matrix
    #         localY = Y + pad_width0
    #         localX = X + pad_width1
    #         imageArea = padded[localY - pad_width0:localY + pad_width0 + 1, localX - pad_width1:localX + pad_width1 + 1]
    #         out[Y, X] = np.sum(np.multiply(imageArea, kernel))


    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = int(size / 2)

    for i in range(size):
        for j in range(size):
            kernel[i, j] = 1 / (2 * np.pi * np.square(sigma)) * np.exp(
                -1 * (np.square(i - k) + np.square(j - k)) / (2 * np.square(sigma)))
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, 0, 0],
                       [-0.5, 0, 0.5],
                       [0, 0, 0]])
    out = conv(img,kernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, -0.5, 0],
                       [0, 0, 0],
                       [0, 0.5, 0]])
    out = conv(img,kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))

    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360

    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    # 只用0,45，90,135 四个方向
    theta = np.floor((theta + 22.5) / 45) * 45 % 180

    ### BEGIN YOUR CODE

    # # 需要注意的是前后不能影响，即非最大抑制的时候，抑制结果不能影响后面的判断，所以处理结果应该放在另外一块存储空间内
    #     #     # # 还是要注意深拷贝和浅拷贝啊
    #     #     # out = G.copy()
    #     #     #
    #     #     # # 计算边缘梯度需要填充边缘
    #     #     # img = np.zeros([H+2,W+2])
    #     #     # img[1:H+1, 1:W+1] = G
    #     #     #
    #     #     # for row in range(H):
    #     #     #     for col in range(W):
    #     #     #         patch = img[row:row+3,col:col+3]
    #     #     #         # print(patch)
    #     #     #
    #     #     #         if theta[row,col] == 0.0:
    #     #     #             if patch[1, 1] != max(patch[1,0], patch[1,1], patch[1,2]):
    #     #     #                 out[row,col] = 0
    #     #     #         elif theta[row,col] == 45.0:
    #     #     #             if patch[1, 1] != max(patch[0,2], patch[1,1], patch[2,0]):
    #     #     #                 out[row, col] = 0
    #     #     #         elif theta[row,col] == 90.0:
    #     #     #             if patch[1, 1] != max(patch[0,1], patch[1,1], patch[2,1]):
    #     #     #                 out[row, col] = 0
    #     #     #         elif theta[row,col] == 135.0:
    #     #     #             if patch[1, 1] != max(patch[0,0], patch[1,1], patch[2,2]):
    #     #     #                 out[row, col] = 0
    #     #     #         else:
    #     #     #             print("Error")

    # 新写的比不上我几个月前写的，哭死
    pos_neg = np.zeros((3, 3))
    pos_neg[1, 1] = 1

    if theta[0, 0] == 0.0:
        pos_neg[1, 0] = 1
        pos_neg[1, 2] = 1
    elif theta[0, 0] == 45.0:
        pos_neg[0, 2] = 1
        pos_neg[2, 0] = 1
    elif theta[0, 0] == 90.0:
        pos_neg[0, 1] = 1
        pos_neg[2, 1] = 1
    elif theta[0, 0] == 135.0:
        pos_neg[0, 0] = 1
        pos_neg[2, 2] = 1
    else:
        print("Error")

    biggerImg = np.zeros((H + 2, W + 2))
    biggerImg[1:H + 1, 1:W + 1] = G

    for Y in range(H):
        for X in range(W):
            localY = Y + 1
            localX = X + 1

            imageArea = biggerImg[localY - 1:localY + 2, localX - 1:localX + 2]

            if imageArea[1, 1] != np.max(np.multiply(imageArea, pos_neg)):
                out[Y, X] = 0
            else:
                out[Y, X] = imageArea[1, 1]
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE

    # # 第一种写法
    # strong_edges = np.zeros(img.shape, dtype=bool)  # add dtype=bool
    # weak_edges = np.zeros(img.shape, dtype=bool)
    # for Y in range(img.shape[0]):
    #     for X in range(img.shape[1]):
    #         if img[Y, X] >= high:
    #             strong_edges[Y, X] = True
    #             weak_edges[Y, X] = False
    #         elif low <= img[Y, X] < high:
    #             strong_edges[Y, X] = False
    #             weak_edges[Y, X] = True
    #         else:
    #             strong_edges[Y, X] = False
    #             weak_edges[Y, X] = False

    # 第二种写法
    strong_edges = img >= high
    weak_edges = (img < high) & (img >= low)

    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ## YOUR CODE HERE
    # for index in indices:
    #     y = index[0]
    #     x = index[1]
    #     edges[y, x] = 1  # label itself
    #
    #     neighbor = get_neighbors(y, x, H, W)
    #     for neighborIndex in neighbor:
    #         neighborY = neighborIndex[0]
    #         neighborX = neighborIndex[1]
    #         if strong_edges[neighborY, neighborX]:
    #             edges[neighborY, neighborX] = 1
    #
    #         if weak_edges[neighborY, neighborX]:
    #             edges[neighborY, neighborX] = 1

    # 之前我这里有个误区就是，weak只能在strong附近才算edge,实际上应该是weak能连通到strong就算edge了
    # 参考mikucy写法
    edges = np.copy(strong_edges)
    for row in range(H):
        for col in range(W):
            neighbors = get_neighbors(row, col, H, W)
            if weak_edges[row, col] and np.any(edges[r, c] for r, c in neighbors):
                edges[row, col] = True

    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i in range(len(ys)):
        x = xs[i]
        y = ys[i]
        for thedaIdx in range(num_thetas):
            rho = x * cos_t[thedaIdx] + y * sin_t[thedaIdx]
            accumulator[int(rho + diag_len), thedaIdx] += 1

    ### END YOUR CODE

    return accumulator, rhos, thetas
