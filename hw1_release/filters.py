import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    #方案一：根据卷积方程定义计算
    for row in range(Hi):
        for col in range(Wi):
            sum = 0
            for i in range(Hk):
                for j in range(Wk):
                    if row+1-i < 0 or col+1-j < 0 or row+1-i >= Hi or col+1-j >= Wi:
                        sum += 0
                    else:
                        sum += kernel[i][j] * image[row+1-i][col+1-j]
            out[row][col] = sum


    # ## 方案二：形象化的卷积操作，通过将卷积核翻转，进行累加的操作。
    # kernel = np.flip(kernel, 0)
    # kernel = np.flip(kernel, 1)
    #
    # center_X = int(Wk / 2)
    # center_Y = int(Hk / 2)
    #
    # for X in range(Wi):
    #     for Y in range(Hi):
    #         for xk in range(Wk):
    #             for yk in range(Hk):
    #                 global_x = X + xk - center_X
    #                 global_y = Y + yk - center_Y
    #
    #                 if 0 <= global_x < Wi and 0 <= global_y < Hi:
    #                     out[Y, X] = out[Y, X] + image[global_y, global_x] * kernel[yk, xk]

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))  # 根据边缘大小设计更大的图像
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image  # 将原图拷贝到新图中心
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # 翻转卷积核
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)

    # 进行零填充
    pad_height = int(Hk / 2)
    pad_width = int(Wk / 2)

    imgBig = zero_pad(image, pad_height, pad_width)  # zero padding

    # 计算像素的输出，此时由于原图附近有零像素点，不需要考虑卷积核越界问题
    for X in range(Wi):
        for Y in range(Hi):
            # trans X,Y to indexs in imgBig matrix
            localX = X + pad_width
            localY = Y + pad_height
            imageArea = imgBig[localY - pad_height:localY + pad_height + 1, localX - pad_width:localX + pad_width + 1]
            out[Y, X] = np.sum(np.multiply(imageArea, kernel))
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # 翻转卷积核
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)

    # 进行零填充
    pad_height = int(Hk / 2)
    pad_width = int(Wk / 2)

    imgBig = zero_pad(image, pad_height, pad_width)  # zero padding

    # 卷积向量化
    kernel_Vec = kernel.reshape(Hk*Wk,1)

    # 计算块向量化，并组成大矩阵
    vec_Mat = np.zeros((Hi*Wi, Hk*Wk))

    for row in range(Hi):
        for col in range(Wi):
            vec_Mat[row*Wi + col, :] = imgBig[row:row+Hk,col:col+Wk].reshape(1, Hk*Wk)

    # 进行内积计算
    result = np.dot(vec_Mat, kernel_Vec)

    # reshape恢复图像尺寸
    out = result.reshape(Hi,Wi)

    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    # 在卷积函数中conv_fast将核进行了翻转，此处将模板翻转是因为计算互相关时匹配核没有反转。
    g = np.flip(g, 0)
    g = np.flip(g, 1)

    out = conv_faster(f,g)

    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = g - g.mean() # g.mean() Or np.mean(g)
    out = cross_correlation(f,g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    out = np.zeros(f.shape)
    # 匹配核做归一化操作
    g = (g - np.mean(g)) / np.var(g)

    # 遍历每个待检测块，进行匹配

    # 进行零填充
    Hg, Wg = g.shape
    Hf, Wf = f.shape

    pad_height = int(Hg / 2)
    pad_width = int(Wg / 2)
    f = zero_pad(f, pad_height, pad_width)  # zero padding

    # 遍历图像块
    # 计算像素的输出，此时由于原图附近有零像素点，不需要考虑卷积核越界问题
    count = 0
    for row in range(Hf):
        for col in range(Wf):
            count = count + 1
            if  count % 200 == 0:
                print(count * 1.0 / (Hf*Wf))
            patch = f[row:row+Hg, col:col+Wg]
            patch = (patch - np.mean(patch))/np.var(patch)
            out[row, col] = np.sum(cross_correlation(patch, g))

    # f = (f - np.mean(f))/np.var(f)
    # out = cross_correlation(f, g)

    ### END YOUR CODE

    return out
