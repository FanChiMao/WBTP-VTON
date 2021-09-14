# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GicLoss, GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")

    parser.add_argument("--name", default="GMM")
    # parser.add_argument("--name", default="TOM")
    parser.add_argument("--stage", default="GMM")
    # parser.add_argument("--stage", default="TOM")

    # parser.add_argument("--data_list", default="train_pairs_cloths_GMM.txt")
    # parser.add_argument("--data_list", default="train_pairs_pants_GMM.txt")
    # parser.add_argument("--data_list", default="train_pairs_cloths_TOM.txt")
    parser.add_argument("--data_list", default="train_pairs_pants_TOM.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=5)
    parser.add_argument("--save_count", type=int, default=1)
    parser.add_argument("--keep_step", type=int, default=1)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
# 006796_0.jpg 000004_1.jpg
    # 015310_0.jpg 000005_1.jpg
    # 016002_0.jpg 000006_1.jpg
    # 016848_0.jpg 000008_1.jpg
    opt = parser.parse_args()
    return opt


def train_gmm(opt, train_loader, model, board):  # GMM
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    gicloss = GicLoss(opt)
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                                                    max(0,
                                                                                        step - opt.keep_step) / float(
        opt.decay_step + 1))
    arr = []
    arr2 = []
    arr3 = []
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        grid, theta = model(agnostic, cm)  # can be added c too for new training
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth + im) * 0.5, im]]

        Lwarp = criterionL1(warped_cloth, im_c)  # loss for warped cloth and pants

        # grid regularization loss
        Lgic = gicloss(grid)
        # 200x200 = 40.000 * 0.001
        Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])

        loss = 0.1 * Lwarp + 40 * Lgic  # total GMM loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            board.add_scalar('loss', loss.item(), step + 1)
            board.add_scalar('40*Lgic', (40 * Lgic).item(), step + 1)
            board.add_scalar('Lwarp', Lwarp.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f, (40*Lgic): %.8f, Lwarp: %.6f' %
                  (step + 1, t, loss.item(), (40 * Lgic).item(), Lwarp.item()), flush=True)
            arr.append(loss.item())
            arr2.append(0.1 * Lwarp.item())
            arr3.append((40 * Lgic).item())

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))
    # x1=range(0,100)
    # y1=arr
    # y2=arr2
    # y3 = arr3
    # print(y1)
    # print(y2)
    # print(y3)
    # plt.plot(x1, y1, 'o-', label="loss")
    # plt.plot(x1, y2, 'o-', label="loss warp")
    # plt.plot(x1, y3, 'o-', label="loss tom")
    # plt.legend()
    # plt.xlabel('steps')
    # plt.ylabel('loss')
    # plt.savefig("accuracy_loss.jpg")
    # plt.show()


def train_tom(opt, train_loader, model, board):  # TOM
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                                                    max(0,
                                                                                        step - opt.keep_step) / float(
        opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        pcm = inputs['parse_cloth_mask'].cuda()

        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        # print((torch.cat([ag c, cm],nostic, 1).expand(1, 24, -1, -1)).shape)
        # padding_data[:, :data_width, :] = data[0] # .repeat(1, 26, 256, 192)
        outputs = model((torch.cat([agnostic, c, cm], 1)))

        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        """visuals = [[im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]"""  # CP-VTON

        visuals = [[im_h, shape, im_pose],
                   [c, pcm * 2 - 1, m_composite * 2 - 1],
                   [p_rendered, p_tryon, im]]

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        # loss_mask = criterionMask(m_composite, cm)  # CP-VTON
        loss_mask = criterionMask(m_composite, pcm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            board.add_scalar('L1', loss_l1.item(), step + 1)
            board.add_scalar('VGG', loss_vgg.item(), step + 1)
            board.add_scalar('MaskL1', loss_mask.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step + 1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'GMM_pant_model.pth'))
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(
            26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'cloth_TOM.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
