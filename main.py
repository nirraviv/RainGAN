import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.gaze_dataset import GazeDataset
from lib.image_history_buffer import ImageHistoryBuffer
from lib.network import Discriminator, Refiner
from lib.image_utils import generate_img_batch, calc_acc
import config as cfg
from pathlib import Path

class Main(object):
    def __init__(self):
        # network
        self.R = None
        self.D = None
        self.opt_R = None
        self.opt_D = None
        self.self_regularization_loss = None
        self.local_adversarial_loss = None
        self.delta = None

        # data
        self.syn_train_loader = None
        self.real_loader = None

        # parameters
        self.device = 'cuda' if cfg.cuda_use else 'cpu'
        self.writer = SummaryWriter(Path('logs') / cfg.experiment_name)
        # initialization flow
        self.build_network()
        self.load_data()
        self.pre_train_refiner()
        self.pre_train_discriminator()

    def build_network(self):
        print('=' * 50)
        print('Building network...')
        self.R = Refiner(4, cfg.img_channels, nb_features=64).to(device=self.device)
        self.D = Discriminator(input_features=cfg.img_channels).to(device=self.device)

        self.opt_R = torch.optim.SGD(self.R.parameters(), lr=cfg.r_lr)
        self.opt_D = torch.optim.SGD(self.D.parameters(), lr=cfg.d_lr)
        self.self_regularization_loss = nn.L1Loss(reduction='sum')
        self.local_adversarial_loss = nn.CrossEntropyLoss(reduction='mean')
        self.delta = cfg.delta

    def load_data(self):
        print('=' * 50)
        print('Loading data...')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        syn_gaze_dataset = GazeDataset(cfg.syn_path, transform=transform)
        self.syn_train_loader = Data.DataLoader(syn_gaze_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                pin_memory=True)
        print(f'# synthetic batches: {len(self.syn_train_loader)}')
        real_gaze_dataset = GazeDataset(cfg.real_path, transform=transform)
        self.real_loader = Data.DataLoader(real_gaze_dataset, batch_size=cfg.batch_size, shuffle=True,
                                           pin_memory=True)
        print(f'# real batches: {len(self.real_loader)}')

    def pre_train_refiner(self):
        print('=' * 50)
        if cfg.ref_pre_path:
            print(f'Loading R_pre from {cfg.ref_pre_path}')
            self.R.load_state_dict(torch.load(cfg.models_path / cfg.ref_pre_path))
            return

        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print(f'pre-training the refiner network {cfg.r_pretrain} times...')

        for index in range(cfg.r_pretrain):
            syn_image_batch = self.syn_train_loader.__iter__().next()
            syn_image_batch = syn_image_batch.to(device=self.device)

            self.R.train()
            ref_image_batch = self.R(syn_image_batch)

            r_loss = self.self_regularization_loss(ref_image_batch, syn_image_batch)
            # r_loss = torch.div(r_loss, cfg.batch_size)
            r_loss = torch.mul(r_loss, self.delta)

            self.opt_R.zero_grad()
            r_loss.backward()
            self.opt_R.step()

            # log every `log_interval` steps
            if (index % cfg.r_pre_per == 0) or (index == cfg.r_pretrain - 1):
                # figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(index)
                print('[{0}/{1}] (R)reg_loss: {2:.4f}'.format(index, cfg.r_pretrain, r_loss.item()))

                syn_image_batch = self.syn_train_loader.__iter__().next()
                syn_image_batch = syn_image_batch.to(device=self.device)

                real_image_batch = self.real_loader.__iter__().next()
                real_image_batch = real_image_batch.to(device=self.device)

                self.R.eval()
                ref_image_batch = self.R(syn_image_batch)

                figure_path = cfg.train_res_path / f'refined_image_batch_pre_train_{index}.png'
                generate_img_batch(syn_image_batch.data.cpu(), ref_image_batch.data.cpu(),
                                   real_image_batch.data, figure_path)
                self.R.train()

                print('Save R_pre to models/R_pre.pkl')
                torch.save(self.R.state_dict(), 'models/R_pre.pkl')

    def pre_train_discriminator(self):
        print('=' * 50)
        if cfg.disc_pre_path:
            print(f'Loading D_pre from {cfg.disc_pre_path}')
            self.D.load_state_dict(torch.load(cfg.models_path / cfg.disc_pre_path))
            return

        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print(f'pre-training the discriminator network {cfg.r_pretrain} times...')

        self.D.train()
        self.R.eval()
        for index in range(cfg.d_pretrain):
            real_image_batch = self.real_loader.__iter__().next()
            real_image_batch = real_image_batch.to(device=self.device)

            syn_image_batch = self.syn_train_loader.__iter__().next()
            syn_image_batch = syn_image_batch.to(device=self.device)

            assert real_image_batch.size(0) == syn_image_batch.size(0)

            # ============ real image D ====================================================
            d_real_pred = self.D(real_image_batch).view(-1, 2)

            d_real_y = d_real_pred.new_zeros(d_real_pred.size(0), dtype=torch.long)
            d_ref_y = torch.ones_like(d_real_y)

            acc_real = calc_acc(d_real_pred, 'real')
            d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
            # d_loss_real = torch.div(d_loss_real, cfg.batch_size)

            # ============ syn image D ====================================================
            ref_image_batch = self.R(syn_image_batch)

            d_ref_pred = self.D(ref_image_batch).view(-1, 2)

            acc_ref = calc_acc(d_ref_pred, 'refine')
            d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
            # d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)

            d_loss = d_loss_real + d_loss_ref
            self.opt_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            if (index % cfg.d_pre_per == 0) or (index == cfg.d_pretrain - 1):
                print('[{0}/{1}] (D)d_loss:{2}  acc_real:{3:.2f}% acc_ref:{4:.2f}%'.format(index, cfg.d_pretrain, d_loss.item(), acc_real*100, acc_ref*100))

        print('Save D_pre to models/D_pre.pkl')
        torch.save(self.D.state_dict(), 'models/D_pre.pkl')

    def train_refiner(self, step):
        self.D.eval()
        self.R.train()

        for p in self.D.parameters():
            p.requires_grad = False

        total_r_loss = 0.0
        total_r_loss_reg_scale = 0.0
        total_r_loss_adv = 0.0
        total_acc_adv = 0.0

        for index in range(cfg.k_r):
            syn_image_batch = self.syn_train_loader.__iter__().next()
            syn_image_batch = syn_image_batch.to(device=self.device)

            ref_image_batch = self.R(syn_image_batch)
            d_ref_pred = self.D(ref_image_batch).view(-1, 2)

            d_real_y = d_ref_pred.new_zeros(d_ref_pred.size(0), dtype=torch.long)

            acc_adv = calc_acc(d_ref_pred, 'real')

            r_loss_reg = self.self_regularization_loss(ref_image_batch, syn_image_batch)
            r_loss_reg_scale = torch.mul(r_loss_reg, self.delta)
            # r_loss_reg_scale = torch.div(r_loss_reg_scale, cfg.batch_size)

            r_loss_adv = self.local_adversarial_loss(d_ref_pred, d_real_y)
            # r_loss_adv = torch.div(r_loss_adv, cfg.batch_size)

            r_loss = r_loss_reg_scale + r_loss_adv

            self.opt_R.zero_grad()
            self.opt_D.zero_grad()
            r_loss.backward()
            self.opt_R.step()

            total_r_loss += r_loss
            total_r_loss_reg_scale += r_loss_reg_scale
            total_r_loss_adv += r_loss_adv
            total_acc_adv += acc_adv
        mean_r_loss = total_r_loss / cfg.k_r
        mean_r_loss_reg_scale = total_r_loss_reg_scale / cfg.k_r
        mean_r_loss_adv = total_r_loss_adv / cfg.k_r
        mean_acc_adv = total_acc_adv / cfg.k_r

        self.writer.add_scalar(tag='refiner/scale_regulizer', scalar_value=mean_r_loss_reg_scale, global_step=step)
        self.writer.add_scalar(tag='refiner/adversarial', scalar_value=mean_r_loss_adv, global_step=step)
        self.writer.add_scalar(tag='acc/refiner', scalar_value=mean_acc_adv, global_step=step)

        print('(R) loss:{0:.4f} loss_reg:{1:.4f}, loss_adv:{2:.4f}({3:.2f}%)'.format(mean_r_loss.item(), mean_r_loss_reg_scale.item(), mean_r_loss_adv.item(), mean_acc_adv*100))

    def train_discriminator(self, image_history_buffer, step):
        self.R.eval()
        self.D.train()
        for p in self.D.parameters():
            p.requires_grad = True

        for index in range(cfg.k_d):
            real_image_batch = self.real_loader.__iter__().next()
            syn_image_batch = self.syn_train_loader.__iter__().next()
            assert real_image_batch.size(0) == syn_image_batch.size(0)

            real_image_batch = real_image_batch.to(device=self.device)
            syn_image_batch = syn_image_batch.to(device=self.device)

            ref_image_batch = self.R(syn_image_batch)

            # use a history of refined images
            half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
            image_history_buffer.add_to_image_history_buffer(ref_image_batch.cpu().data.numpy())

            if len(half_batch_from_image_history):
                torch_type = torch.from_numpy(half_batch_from_image_history)
                v_type = torch_type.to(device=self.device)
                ref_image_batch[:cfg.batch_size // 2] = v_type

            d_real_pred = self.D(real_image_batch).view(-1, 2)

            d_real_y = d_real_pred.new_zeros(d_real_pred.size(0), dtype=torch.long)
            d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
            # d_loss_real = torch.div(d_loss_real, cfg.batch_size)
            acc_real = calc_acc(d_real_pred, 'real')

            d_ref_pred = self.D(ref_image_batch).view(-1, 2)
            d_ref_y = d_real_pred.new_ones(d_ref_pred.size(0), dtype=torch.long)
            d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
            # d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)
            acc_ref = calc_acc(d_ref_pred, 'refine')

            d_loss = d_loss_real + d_loss_ref

            self.D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            self.writer.add_scalar(tag='discriminator/real', scalar_value=d_loss_real, global_step=step)
            self.writer.add_scalar(tag='discriminator/adversarial', scalar_value=d_loss_ref, global_step=step)
            self.writer.add_scalar(tag='acc/discriminator_real', scalar_value=acc_real, global_step=step)
            self.writer.add_scalar(tag='acc/discriminator_refiner', scalar_value=acc_ref, global_step=step)

            print('(D) loss:{0:.4f} real_loss:{1:.4f}({2:.2f}%) refine_loss:{3:.4f}({4:.2f}%)'.format(d_loss.item() / 2, d_loss_real.item(), acc_real*100, d_loss_ref.item(), acc_ref*100))

    def train(self):
        print('=' * 50)
        print('Training...')
        image_history_buffer = ImageHistoryBuffer((0, cfg.img_channels, cfg.img_height, cfg.img_width),
                                                  cfg.buffer_size * 10, cfg.batch_size)
        for step in range(cfg.train_steps):
            print('Step[%d/%d]' % (step, cfg.train_steps))

            self.train_refiner(step=step)

            self.train_discriminator(image_history_buffer, step=step)

            if step % cfg.save_per == 0:
                print('Save two model dict.')
                torch.save(self.D.state_dict(), cfg.models_path / cfg.D_path.format(step))
                torch.save(self.R.state_dict(), cfg.models_path / cfg.R_path.format(step))

                with torch.no_grad():
                    real_image_batch = self.real_loader.__iter__().next()
                    syn_image_batch = self.syn_train_loader.__iter__().next()
                    real_image_batch = real_image_batch.to(device=self.device)
                    syn_image_batch = syn_image_batch.to(device=self.device)

                    self.R.eval()
                    ref_image_batch = self.R(syn_image_batch)
                    self.generate_batch_train_image(syn_image_batch, ref_image_batch, real_image_batch, step_index=step)

    def generate_batch_train_image(self, syn_image_batch, ref_image_batch, real_image_batch, step_index=-1):
        print('=' * 50)
        print('Generating a batch of training images...')
        self.R.eval()

        pic_path = cfg.train_res_path / f'step_{step_index}.png'
        img = generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, pic_path)
        self.writer.add_image('Images', img, global_step=step_index, dataformats='HWC')
        print('=' * 50)


if __name__ == '__main__':
    obj = Main()
    obj.train()


