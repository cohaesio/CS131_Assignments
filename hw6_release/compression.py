import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size

    # 奇异值分解
    u,s,v = np.linalg.svd(image)

    # 保留前n个奇异值和前n个左奇异向量和右奇异向量
    u = u[:, :num_values]
    s = s[:num_values]
    v = v[:num_values,:]

    # 重新构建矩阵,需要将奇异值序列构建成对角矩阵
    compressed_image = np.dot(np.dot(u, np.diag(s)),v)

    # 计算压缩的size,什么意思呢？
    compressed_size = image.shape[0] * u.shape[1] + num_values + v.shape[0] * image.shape[1]
    
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
