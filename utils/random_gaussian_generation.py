import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# # Configuration
# NUMBER_OF_IMAGES = 50000
# DIM = [35, 55]
# PIXEL_SIZE = 2
# SAVE_DIR = Path('dataset')
# OFFSET = -30
# GAIN = 0.4


def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array pos.
    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N


def create_one_random_gauss(cfg):
    dim = cfg.image_size
    mean_x = np.random.uniform(0.1 * dim[0], 0.9 * dim[0])
    mean_y = np.random.uniform(0.1 * dim[1], 0.9 * dim[1])
    cov_x = np.random.uniform(0.1 * dim[0], 0.5 * dim[0])
    cov_y = np.random.uniform(0.1 * dim[1], 0.5 * dim[1])
    teta = np.random.uniform(0, 2 * np.pi)
    rotation = np.array([[np.cos(teta), -np.sin(teta)], [np.sin(teta), np.cos(teta)]])
    cov = rotation @ np.array([[cov_x ** 2, 0], [0, cov_y ** 2]]) @ rotation.T
    mag = np.random.uniform(0.1, 100) # R = rainfall [mm/h]

    x = np.arange(dim[1])
    y = np.arange(dim[0])
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D

    pos = np.stack([x, y], -1)
    mean = np.array([mean_x, mean_y])

    z = multivariate_gaussian(pos, mean, cov)

    z = z / z.max() * mag
    return z


def create_number_of_gaussian(number, cfg):
    ''' nunber -  Numver of random gausses to return'''
    gausses = []
    with tqdm(total=number) as pbar:
        for i in range(number):
            tmp = create_one_random_gauss(cfg)
            gausses.append(rain2pixel(tmp, cfg))
            pbar.update(1)
    return gausses


def rain2pixel(r, cfg):
    """
    z = 10log10(200*r**1.5)
    dBZ = pixel value * Gain + Offset
    :param r:
    :return:
    """
    offset, gain = cfg.z_to_pix
    z = 10*np.log10(200*r**1.5)
    z[z > 70] = 70  # too high
    pixel = (z - offset)/gain
    pixel[r < 0.06] = 0
    return pixel.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create synthetic images')
    parser.add_argument('--save_path', '-s', type=Path, default=Path("dataset"),
                        help='save directory path')
    parser.add_argument('--image_size', type=tuple, default=(35,55),
                        help='image dimensions (H,W)')
    parser.add_argument('--pixel_size', type=int, default=2,
                        help='size of each pixel (km)')
    parser.add_argument('--num_images', type=int, default=100000,
                        help='number of images to generate')
    parser.add_argument('--z_to_pix', type=tuple, default=(-30, 0.4),
                        help='z  to pixel parameters (offset, gain)')
    args = parser.parse_args()

    imgs = create_number_of_gaussian(args.num_images, args)
    imgs = np.expand_dims(np.stack(imgs), -1)
    np.save(args.save_path / 'syn_rain.npy', imgs)
    print(f'Done {imgs.shape[0]} synthetic images')
