import numpy as np
import matplotlib.image as im_mpl
import sys


def get_svd(mat):
    return np.linalg.svd(mat)


def next_unfolding(mat, dimension_product, new_dim):
    """
    Get A_(n + 1) tensor unfolding based on A_(n).
    On the same column in A_(n) all indices except i_(n) are the same,
    thus we can write them consequently in a row of A_(n+1).
    """
    h = new_dim
    w = dimension_product // new_dim
    mat_new = np.zeros((h, w))
    for i in range(mat.shape[1]):
        new_col = (i * mat.shape[0]) % w
        mat_new[(i * mat.shape[0]) // w, new_col : new_col + mat.shape[0]] = np.array(mat[:, i])
    return mat_new


def get_frobenius_norm(mat):
    flat = mat.flat
    res = 0
    for el in flat:
        res += el.conjugate() * el
    return np.sqrt(res)


sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')
np.set_printoptions(4)


#---Read image-----------------------------------------------------------
input_name = input()
output_name = input()
compression_rate = int(input())

img = im_mpl.imread(input_name)


#---Configure input-------------------------------------------------------

dim_am = 3  # Read how many dimensions the tensor has
dim = np.asarray(img.shape)
dimension_product = dim.prod()
unfolded_tensor = np.reshape(img.flat, (dim[0], dimension_product // dim[0]))


#---Find U^(1), U^(2), ..., U^(dim_am) matrices, use equation 15----------

U = []  # All U matrices
for i in range(dim_am):
    U.append(np.linalg.svd(unfolded_tensor)[0])
    unfolded_tensor = next_unfolding(unfolded_tensor, dimension_product, dim[(i + 1) % dim_am])

unfolded_res = U[0].T @ unfolded_tensor  # For complex numbers conjugate after .T is required
kron_prod = U[1]
for i in range(2, dim_am):
    kron_prod = np.kron(kron_prod, U[i])
unfolded_res = unfolded_res @ kron_prod

print('original size:', img.size)
print('basic HOSVD size:', unfolded_res.size + sum(u.size for u in U))


#---Dismiss all sigma^(1) values---------------------------------------------------

for i in range(2):
    dim[i] //= compression_rate
    dimension_product = dim.prod()

    U[i] = U[i][:, :dim[i]]
    unfolded_res = unfolded_res[:dim[i], :]

    unfolded_res = next_unfolding(unfolded_res, dimension_product, dim[(i + 1) % dim_am])

unfolded_res = next_unfolding(unfolded_res, dimension_product, dim[0])

print('HOSVD size after compression:', unfolded_res.size + sum(u.size for u in U))


#---Computing new tensor using formula (15)-------------------------------------------

kron_prod = U[1]
for i in range(2, dim_am):
    kron_prod = np.kron(kron_prod, U[i])

img_res = U[0] @ unfolded_res @ kron_prod.T  # For complex case conjugate after .T is required
img_res = np.clip(img_res, 0, 1)

#---Saving image----------------------------------------------------------------------

img_res = np.reshape(img_res, img.shape)
im_mpl.imsave(output_name, np.reshape(img_res, img.shape))