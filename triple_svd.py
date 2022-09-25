import numpy as np


def triple_svd(tensor, compression_rate):
    dim = np.asarray(tensor.shape)
    cut_dim = dim.copy()
    for i in range(2):
        cut_dim[i] //= compression_rate

    # ---Split tensor into several 2D matrices and compress each-------------------------

    res = []  # List of tuples for SVDs' of each color matrix
    for i in range(dim[2]):  # Split tensor into 3 2D matrices representing each color
        full_svd = np.linalg.svd(tensor[:, :, i])
        u_svd = full_svd[0][:, :dim[0] // compression_rate]
        s_svd = full_svd[1][:min(dim[0] // compression_rate, dim[1] // compression_rate)]
        v_svd = full_svd[2][:dim[1] // compression_rate, :]
        res.append((u_svd, s_svd, v_svd))

    triple_svd_size = 0
    for i in range(dim[2]):
        triple_svd_size += res[i][0].size + res[i][1].size + res[i][2].size

    print('Triple SVD size after compression:', triple_svd_size)
    return res


def restore_tensor(comp_tensor, dim):
    # ---Computing new tensor from triple SVD compression and restoring image---------------------------------

    color_matrices_restored = []
    for i in range(dim[2]):
        color_matrices_restored.append(comp_tensor[i][0] @ np.diag(comp_tensor[i][1]) @ comp_tensor[i][2])

    res = np.zeros(dim)
    for i in range(dim[2]):
        res[:, :, i] = color_matrices_restored[i]
    return np.reshape(res, dim)
