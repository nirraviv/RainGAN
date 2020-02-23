import numpy as np
from PIL import Image
import torch
import config as cfg


def restore_img(img):
    # img_type: numpy
    img += 1
    img /= 2
    img *= 255
    img = img.astype(np.uint8)
    return img


def generate_img_batch(syn_batch, ref_batch, real_batch, png_path=None):
    # syn_batch_type: Tensor, ref_batch_type: Tensor
    def tensor_to_numpy(img):
        img = restore_img(img.cpu().numpy())
        img = np.transpose(img, [1, 2, 0])
        return img

    syn_batch = syn_batch[:64]
    ref_batch = ref_batch[:64]
    real_batch = real_batch[:64]

    a_blank = torch.zeros(cfg.img_height, cfg.img_width*2, 1).numpy().astype(np.uint8)

    nb = syn_batch.size(0)
    vertical_list = []

    for index in range(0, nb, cfg.pics_line):
        st = index
        end = st + cfg.pics_line

        if end > nb:
            end = nb

        syn_line = syn_batch[st:end]
        ref_line = ref_batch[st:end]
        real_line = real_batch[st:end]
        nb_per_line = syn_line.size(0)

        line_list = []

        for i in range(nb_per_line):
            syn_np = tensor_to_numpy(syn_line[i])
            ref_np = tensor_to_numpy(ref_line[i])
            real_np = tensor_to_numpy(real_line[i])
            a_group = np.concatenate([syn_np, ref_np, real_np], axis=1)
            line_list.append(a_group)

        fill_nb = cfg.pics_line - nb_per_line
        while fill_nb:
            line_list.append(a_blank)
            fill_nb -= 1
        line = np.concatenate(line_list, axis=1)
        vertical_list.append(line)

    imgs = np.concatenate(vertical_list, axis=0)
    if imgs.shape[-1] == 1:
        imgs = np.tile(imgs, [1, 1, 3])

    if png_path is not None:
        img = Image.fromarray(imgs)
        img.save(png_path, 'png')

    return imgs


def calc_acc(output, type='real'):
    assert type in ['real', 'refine']

    if type == 'real':
        label = output.new_zeros(output.size(0), dtype=torch.long)
    else:
        label = output.new_ones(output.size(0), dtype=torch.long)

    if output.ndim > 1:
        softmax_output = torch.softmax(output, dim=1)
        acc = (softmax_output.max(1)[1] == label).cpu().numpy()

    else:
        softmax_output = torch.sigmoid(output)
        acc = (torch.round(softmax_output) == label).cpu().numpy()

    return acc.mean()
