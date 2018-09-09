import numpy as np
from skimage import color


def energy_function(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: you can use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W, _ = image.shape
    out = np.zeros((H, W))

    ### YOUR CODE HERE
    grayImg = color.rgb2gray(image)


    gradient = np.gradient(grayImg)
    out = np.abs(gradient[0]) + np.abs(gradient[1])

    # dx = np.gradient(grayImg, axis=0)
    # dy = np.gradient(grayImg, axis=1)
    # out = np.abs(dx) + np.abs(dy)
    ### END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    energy = energy.copy()

    # 如果是实现裁剪高度，则将图像旋转，
    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    # 要用一个循环实现，则循环的只有行数了
    for row in range(1, H):
        # 要直接在上一行中实现检索左中右三个的最小值不太现实
        # 可以构建三通道的值，每个通道分别代表了上左，上中，上右
        upL = np.insert(cost[row-1, 0:W-1], 0, 1e10, axis=0)
        # print(upL)
        upM = cost[row-1,:]
        # print(upM)
        upR = np.insert(cost[row-1, 1:W], W-1, 1e10, axis=0)
        # print(upR)
        # 拼接可以使用np.concatenate，但是np.r_或np.c_更高效
        # upchoices = np.r_[upL, upM, upR].reshape(3, -1)
        upchoices=np.concatenate((upL, upM,upR), axis=0).reshape(3,-1)
        cost[row] = energy[row] + np.min(upchoices,axis=0)
        paths[row] = np.argmin(upchoices, axis=0) - 1   #-1,0,1分别表示左中右
    ### END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    seam = np.zeros(H, dtype=np.int)

    # Initialization
    seam[H-1] = end

    ### YOUR CODE HERE
    for row in range(H-2, -1, -1):
        seam[row] = seam[row+1] + paths[row+1, seam[row+1]]
    ### END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
    """

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    ### YOUR CODE HERE
    # 根据seam和原有图像尺寸构造新的图像像素区域
    # 方式一：没理解
    # out = image[np.arange(W) != seam[:, None]].reshape(H, W - 1, C)

    # 方式二：每行删除一个
    out = np.zeros((H,W-1,C))
    for i in range(H):
        out[i] = np.delete(image[i], seam[i], axis=0)

    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    # 缩减宽度至目标宽度
    while out.shape[1] > size:
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)
        out = remove_seam(out, seam)
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".

    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    # 每行添加一个像素
    out = np.zeros((H,W+1,C))
    for i in range(H):
        out[i] = np.insert(image[i], seam[i],image[i,seam[i]], axis=0)
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    # 夸大宽度至目标宽度
    while out.shape[1] < size:
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)
        out = duplicate_seam(out, seam)
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    # 创造一个能够记住原有像素下标的矩阵,这样每次获取的接缝路径值都是在原图上的下标
    # np.tile函数是进行复制，每行（1，H）复制H行。
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    # 用来保存接缝路径,序号就表示了第几条路径
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        # 图像中移除接缝，进而进行下一条接缝的判断
        image = remove_seam(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        # 没问题啊，indices[np.arange(H), seam]表示此次接缝的横轴坐标，seams[np.arange(H), indices[np.arange(H), seam]]表示此次接缝坐标在seams图中所在位置的值。
        # 我知道了问题出在remove_seam函数上，默认构造函数为float，indices就被指定为float了，需要指定为int类型
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]]) == 0, \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        # indices也删除改天接缝，图像大小和正在检测image维持同样大小，但是保存的值是相对于最开始的源图的像素下标的。
        indices = remove_seam(indices, seam).astype(int)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    Use functions:
        - find_seams
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    ### YOUR CODE HERE
    # print(out.shape)
    seamsNum = size - W
    seams = find_seams(out, seamsNum)
    # print(seams.shape)
    # 需要将seams转换为三维的
    # 否则无法进行duplicate函数操作
    seams = seams[:,:,np.newaxis]
    # print(seams.shape)
    for i in range(seamsNum):
        seam = np.where(seams == i+1)[1]

        out = duplicate_seam(out, seam)

        # 需要保持和out维度一致才不会引起坐标混乱
        seams = duplicate_seam(seams, seam)

    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):
    """Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j+1] - image[0, j-1])
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    for row in range(1,H):
        # 先获取之前的像素相邻三个能量值
        upL = np.insert(cost[row - 1, 0:W - 1], 0, 1e10, axis=0)
        upM = cost[row - 1, :]
        upR = np.insert(cost[row - 1, 1:W], W - 1, 1e10, axis=0)
        # 拼接可以使用np.concatenate，但是np.r_或np.c_更高效
        # upchoices = np.r_[upL, upM, upR].reshape(3, -1)
        # upchoices = np.concatenate((upL, upM, upR), axis=0).reshape(3, -1)

        # I(i,j+1)
        I_i_j_P = np.insert(image[row,0:W-1],0,0,axis=0)
        # I(i,j-1)
        I_i_j_M = np.insert(image[row,1:W],W-1,0,axis=0)
        # I(i-1.j)
        I_M = image[row-1,:]

        C_V = abs(I_i_j_P - I_i_j_M)
        C_V[0] = 0
        C_V[-1] = 0

        C_L = C_V + abs(I_M - I_i_j_P)
        C_L[0] =0

        C_R =C_V + abs(I_M - I_i_j_M)
        C_R[-1] = 0

        upchoices = np.concatenate((upL+C_L, upM+C_V, upR+C_R), axis=0).reshape(3, -1)

        cost[row] = energy[row] + np.min(upchoices,axis=0)
        paths[row] = np.argmin(upchoices, axis=0) - 1   #-1,0,1分别表示左中右
    ### END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Hint: do we really need to compute the whole cost map again at each iteration?

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    # # mikucy阶梯解题方法
    # energy = efunc(out)
    # while out.shape[1] > size:
    #     cost, paths = cfunc(out, energy)
    #     end = np.argmin(cost[-1])
    #     seam = backtrack_seam(paths, end)
    #     # Get the seam area
    #     i = np.min(seam)
    #     j = np.max(seam)
    #     out = remove_seam(out, seam)
    #     if i <= 3:
    #         energy = np.c_[efunc(out[:, 0: j + 2])[:, : -1], energy[:, j + 2:]]
    #     elif j >= out.shape[1] - 3:
    #         energy = np.c_[energy[:, 0: i - 1], efunc(out[:, i - 3:])[:, 2:]]
    #     else:
    #         energy = np.c_[energy[:, 0: i - 1], efunc(out[:, i - 3: j + 2])[:, 2: -1], energy[:, j + 2:]]


    # 就是计算energy能量图这一步，因为能量图的计算其实就是梯度的计算。
    # 如果一个区域内的图像像素不变动的话，那这个区域内的梯度应该也不会放生改变，基于这一点，可以对计算能量图的步骤进行改进。
    # 只需要对上一次optical seam最小能量线覆盖宽度内的区域进行梯度的重新更新，两侧区域的能量值不需要变动。

    energy = efunc(out)
    while out.shape[1] > size:
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)
        out = remove_seam(out, seam)

        # 获取区域宽度
        left = np.min(seam)
        right = np.max(seam)

        # # left-3和right+2是考虑到虽然从left-1之前的和right之后的元素梯度是不变的。但是区域内的梯度却要从边缘再向外拓展两个像素再计算再取区域内像素
        # # 当然也要考虑到特殊情况，避免数组越界了
        if left <= 3:
            energy = np.c_[efunc(out[:, 0: right + 2])[:, : -1], energy[:, right + 2:]]
        elif right >= out.shape[1] - 3:
            energy = np.c_[energy[:, 0: left - 1], efunc(out[:, left - 3:])[:, 2:]]
        else:
            energy = np.c_[energy[:, 0: left - 1], efunc(out[:, left - 3: right + 2])[:, 2: -1], energy[:, right + 2:]]
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    out = np.copy(image)

    ### YOUR CODE HERE
    # 参考过mikucy的做法，解决方案中还包括了统计删除对象个数的判别和删除对象的长宽之比，可能是为了尽可能的删除最少的能量线吧。

    # 此处我不考虑上面两种拓展方式，只考虑将对象移除
    H,W,C = out.shape
    while not np.all(mask == 0):
        energy = energy_function(out)
        weighted_energy = energy + mask * (-100)
        cost, paths = compute_forward_cost(out, weighted_energy)
        end = np.argmin(cost[-1])

        seam = backtrack_seam(paths, end)
        out = remove_seam(out, seam)
        mask = remove_seam(mask,seam)

    # 再将图像扩大的原来的大小
    out = enlarge(out, W, axis=1)

    ### END YOUR CODE

    return out
