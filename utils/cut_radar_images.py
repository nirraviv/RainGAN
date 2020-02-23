import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

# # Folder config:
# RADAR_IMGES_FOLDER = Path("RainRealImages")
# SAVE_FOLDER = Path("dataset")
#
# # Images size Config:
# IMAGE_PIXELS_SIZE = (35, 55)  # image dimensions (H,W)
# NUMBER_OF_PIXELS = IMAGE_PIXELS_SIZE[0] * IMAGE_PIXELS_SIZE[1]
# NOT_ZEROES_IN_CROPPED_IMAGE = 0.5
#
# # Cutting rectangular data
# STEP_DOWN = 20
# STEP_RIGHT = 25
# LeftStart = 70  # 240
# LeftEnd = 360
# UpperStart = 600  # 50
# UppEnd = 700  # 180


def cut_img(im, top_left_corner, cfg):
    ''' crop_rectangle == (left, upper, right, lower)'''
    # parse cfg
    image_size = cfg.size
    none_zero_per = cfg.cloud

    # start crop
    col, row = top_left_corner
    cropped_im = im[row:row+image_size[0], col:col+image_size[1]]
    unique = np.unique(cropped_im)
    if 255 in unique: return
    num_non_zero = np.count_nonzero(cropped_im)
    if num_non_zero < none_zero_per * image_size[0] * image_size[1]:
        return
    return np.expand_dims(cropped_im, -1)  # add channel


def cut_all_valid_images(im, imgs, cfg):
    ''' Cut the all valid images in the upper circle in given Image '''
    # parse args
    left_start, up_start = cfg.top_left
    left_end, up_end = cfg.bottom_right
    image_size = cfg.size

    # Cuttint loop:
    up = up_start
    while up < up_end:
        left = left_start
        while left < left_end:
            top_left_point = [left, up]
            cropped_im = cut_img(im, top_left_point, cfg)
            if cropped_im is not None:
                imgs.append(cropped_im)
            left += image_size[0]
            #####
        up += cfg.step_down
        #####
    # End of cutting loop
    return imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess real radar images from  downloaded from SHMI')
    parser.add_argument('--image_path', '-i', type=Path, default=Path("RainRealImages"),
                        help='Radar images path directory')
    parser.add_argument('--save_path', '-s', type=Path, default=Path("dataset"),
                        help='Save directory path')
    parser.add_argument('--size', type=tuple, default=(35,55),
                        help='image dimensions (H,W)')
    parser.add_argument('--cloud', type=float, default=0.3,
                        help='percentage of rain pixels out of the total image size (divided by 100)')
    parser.add_argument('--step_down', type=int, default=20,
                        help='number of pixels to move down the crop window')
    parser.add_argument('--step_right', type=int, default=25,
                        help='number of pixels to move right the crop window')
    parser.add_argument('--top_left', type=tuple, default=(70, 600),
                        help='tuple for top left corner (pixels)')
    parser.add_argument('--bottom_right', type=tuple, default=(360, 700),
                        help='tuple for bottom right corner (pixels)')

    args = parser.parse_args()

    # Open Radar File image:
    data_folder = args.image_path
    all_imgs = []
    all_files = list(data_folder.glob('*.tif'))
    for filename in tqdm(all_files):
        image_np = np.array(Image.open(filename))
        all_imgs = cut_all_valid_images(image_np, all_imgs, args)

    all_imgs = np.stack(all_imgs)
    np.save(args.save_path / 'real_rain_new.npy', all_imgs)

    print(f"Done {all_imgs.shape[0]} images!!")


