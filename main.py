import numpy as np
import matplotlib.image as im_mpl
import hosvd_img
import triple_svd
import sys

sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')
np.set_printoptions(4)

# ---Read image-----------------------------
input_name = input()
output_name = input()
compression_rate = int(input())

img = im_mpl.imread(input_name)

# ---HOSVD compress/restore-----------------
compressed_img, U = hosvd_img.get_img_hosvd(img, compression_rate)

img_restored = hosvd_img.restore_img_tensor(compressed_img, U, img.shape)
img_restored = np.clip(img_restored, 0, 1)
im_mpl.imsave('hosvd_' + output_name, np.reshape(img_restored, img.shape))

# ---Triple SVD compress/restore------------
compressed_img = triple_svd.triple_svd(img, compression_rate)

img_restored = triple_svd.restore_tensor(compressed_img, img.shape)
img_restored = np.clip(img_restored, 0, 1)
im_mpl.imsave('triple_svd_' + output_name, np.reshape(img_restored, img.shape))


