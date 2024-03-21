import torch
torch.set_printoptions(10)

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter

from models import squid
from models import hierarchical_memory
from models.memory import MemoryQueue

import random
import importlib

from tqdm import tqdm

from tools import parse_args, build_disc, log, log_loss, save_image, backup_files
from alert import GanAlert


args = parse_args()

CONFIG = importlib.import_module('configs.'+args.config).Config()

if not os.path.exists(os.path.join('checkpoints', args.exp)):
    os.makedirs(os.path.join('checkpoints', args.exp), exist_ok=True)

if not os.path.exists(os.path.join('checkpoints', args.exp, 'test_images')):
    os.makedirs(os.path.join('checkpoints', args.exp, 'test_images'), exist_ok=True)

save_path = os.path.join('checkpoints', args.exp, 'test_images')

# log
log_file = open(os.path.join('checkpoints', args.exp, 'log.txt'), 'w')
writer = SummaryWriter(log_dir=os.path.join('checkpoints', args.exp))

# backup files
backup_files(args)

# main AE
model = squid.AE(CONFIG, 32, level=CONFIG.level).cuda()
opt = CONFIG.opt(model.parameters(), lr=CONFIG.lr, eps=1e-7, betas=(0.5, 0.999), weight_decay=0.00001)
scheduler = CONFIG.scheduler(opt, **CONFIG.scheduler_args)

# for discriminator
if (CONFIG.enbale_gan is not None and CONFIG.enbale_gan >= 0):
    discriminator = build_disc(CONFIG).cuda()
    opt_d = CONFIG.opt(discriminator.parameters(), betas=(0.5, 0.999), lr=CONFIG.gan_lr)
    # scheduler_d = CONFIG.scheduler_d(opt_d, **CONFIG.scheduler_args_d)

# criterions
ce = nn.BCEWithLogitsLoss().cuda()
recon_criterion = torch.nn.MSELoss(reduction='mean').cuda()

# alert
alert = GanAlert(discriminator=discriminator, args=args, CONFIG=CONFIG, generator=model)


def main():

    best_auc = -1
    best_epoch = 0

    for epoch in range(1, CONFIG.epochs + 1):
        start_time = time.time()
        # when GAN training is disabled
        if CONFIG.enbale_gan is None or epoch < CONFIG.enbale_gan:
            train_loss = train(CONFIG.train_loader, epoch)
            val_loss = {'recon_l1': 0.}
            log_loss(log_file, epoch, train_loss, val_loss)
            continue

        # when GAN training is enabled
        train_loss = gan_train(CONFIG.train_loader, epoch, writer)
        print(time.time() - start_time)
        # do we need scheduler for discriminator?
        scheduler.step()

        if epoch % CONFIG.val_freq == 0:
            train_recon_scores, train_real_scores = alert.collect(CONFIG.train_loader)
            reconstructed, inputs, scores, labels, val_loss = val(CONFIG.val_loader, epoch)
            val_recon_scores, val_real_scores = alert.collect(CONFIG.val_loader)

            log_loss(log_file, epoch, train_loss, val_loss)

            # alert, collect=true uses train set mean/std
            results = alert.evaluate(scores, labels, collect=True)

            # log metrics
            msg = '[VAL metrics] '
            for k, v in results.items():
                if np.isscalar(v):
                    msg += k + ': '
                    msg += '%.2f ' % v
            log(log_file, msg)

            for loss_name in train_loss:
                writer.add_scalar(f'Loss/{loss_name}/train', train_loss[loss_name], epoch)
            writer.add_histogram('Discriminator/train_recon_scores', train_recon_scores, epoch)
            writer.add_histogram('Discriminator/train_real_scores', train_real_scores, epoch)
            writer.add_histogram('Discriminator/val_recon_scores', val_recon_scores, epoch)
            writer.add_histogram('Discriminator/val_real_scores', val_real_scores, epoch)
            val_normal_scores = np.array([scores[i] for i in range(len(scores)) if labels[i] == 0])
            val_abnormal_scores = np.array([scores[i] for i in range(len(scores)) if labels[i] == 1])
            writer.add_histogram('Discriminator/val_normal_mean', val_normal_scores, epoch)
            writer.add_histogram('Discriminator/val_abnormal_mean', val_abnormal_scores, epoch)
            for metric_name in results:
                if np.isscalar(results[metric_name]):
                    writer.add_scalar(f'Metric/{metric_name}/val', results[metric_name], epoch)

            # save best model
            if results['auc'] > best_auc:
                best_auc = results['auc']
                best_epoch = epoch
                save_image(os.path.join(save_path, 'best'), zip(reconstructed, inputs))
                if CONFIG.enbale_gan is not None:
                    torch.save(discriminator.state_dict(), os.path.join('checkpoints',args.exp,'discriminator.pth'))
                torch.save(model.state_dict(), os.path.join('checkpoints',args.exp,'model.pth'))
                log(log_file, 'save model!')

        else:
            log_loss(log_file, epoch, train_loss, {})

        if epoch % 50 == 0:
            if CONFIG.enbale_gan is not None:
                torch.save(discriminator.state_dict(), os.path.join('checkpoints', args.exp, f'discriminator_epoch{epoch}.pth'))
            torch.save(model.state_dict(), os.path.join('checkpoints', args.exp, f'model_epoch{epoch}.pth'))

        # save_image(os.path.join(save_path, 'last'), zip(reconstructed, inputs))

        # save last 10 epochs generated imgs for debugging
        # if epoch >= CONFIG.epochs - 10:
        #     save_image(os.path.join(save_path, 'epoch_'+str(epoch)), zip(reconstructed, inputs))

    print(f'Best epoch: {best_epoch}, best auc {best_auc}')

    log_file.close()
    writer.close()


def train(dataloader, epoch):
    model.train()
    batches_done = 0
    tot_loss = {'recon_loss': 0., 'g_loss': 0., 'd_loss': 0., 't_recon_loss': 0., 'dist_loss': 0.}

    # clip dataloader
    if CONFIG.limit is None:
        limit = len(dataloader) - len(dataloader) % CONFIG.n_critic
    else:
        limit = CONFIG.limit

    for i, (img, label) in enumerate(tqdm(dataloader, disable=CONFIG.disable_tqdm)):
        if i > limit:
            break
        batches_done += 1

        img = img.to(CONFIG.device)
        label = label.to(CONFIG.device)

        opt.zero_grad()

        out = model(img)

        if CONFIG.alert is not None:
            CONFIG.alert.record(out['recon'].detach(), img)

        loss_all = CONFIG.recon_w * recon_criterion(out["recon"], img)
        tot_loss['recon_loss'] += loss_all.item()

        if CONFIG.dist and 'teacher_recon' in out and torch.is_tensor(out['teacher_recon']):
            t_recon_loss = CONFIG.t_w * recon_criterion(out["teacher_recon"], img)
            loss_all =  loss_all + t_recon_loss
            tot_loss['t_recon_loss'] += t_recon_loss.item()

        if  CONFIG.dist and 'dist_loss' in out and torch.is_tensor(out['dist_loss']):
            dist_loss = CONFIG.dist_w  * out["dist_loss"]
            loss_all = loss_all + dist_loss
            tot_loss['dist_loss'] += dist_loss.item()

        loss_all.backward()
        opt.step()

        for module in model.modules():
            if isinstance(module, MemoryQueue):
                module.update()

    # avg loss
    for k, v in tot_loss.items():
        tot_loss[k] /= batches_done

    return tot_loss

def gan_train(dataloader, epoch, writer):
    model.train()
    batches_done = 0
    tot_loss = {'loss': 0., 'recon_loss': 0., 'g_loss': 0., 'd_loss': 0., 't_recon_loss': 0., 'dist_loss': 0.}

    # clip dataloader
    if CONFIG.limit is None:
        limit = len(dataloader) - len(dataloader) % CONFIG.n_critic
    else:
        limit = CONFIG.limit

    # progressive learning and fade in skip connection to avoid lazy model
    fadein_weights = [0.0 for _ in range(CONFIG.level)]
    if epoch > 200:
        fadein_weights[-1] += min((epoch - 200) / (400 - 200), 1.0)
    if epoch > 400:
        fadein_weights[-2] += min((epoch - 400) / (600 - 400), 1.0)
    fadein_weights = [0.0, 0.0, 1.0, 1.0]
    print(f'Epoch {epoch}, fadein weights {fadein_weights}')

    for i, (img, label) in enumerate(tqdm(dataloader, disable=CONFIG.disable_tqdm)):
        if i > limit:
            break
        batches_done += 1
        iter_start = time.time()

        img = img.to(CONFIG.device)
        label = label.to(CONFIG.device)

        opt_d.zero_grad()
        out = model(img, fadein_weights=fadein_weights)

        ae_time =time.time()

        # Real images
        real_validity = discriminator(img)
        # Fake images
        fake_validity = discriminator(out["recon"].detach())

        disc_time = time.time()

        # cross_entropy loss
        d_loss = ce(real_validity, torch.ones_like(real_validity))
        d_loss += ce(fake_validity, torch.zeros_like(fake_validity))
        d_loss *= CONFIG.d_w
        d_loss.backward()
        summary_grads(discriminator, 'discriminator', writer, epoch)
        opt_d.step()

        tot_loss['d_loss'] += d_loss.item()

        disc_bw_time = time.time()

        # train generator at every n_critic step only
        if i % CONFIG.n_critic == 0:

            # out = model(img)

            if CONFIG.alert is not None:
                CONFIG.alert.record(out['recon'].detach(), img)

            # reconstruction loss
            recon_loss = CONFIG.recon_w * recon_criterion(out["recon"], img)
            tot_loss['recon_loss'] += recon_loss.item()
            loss_all = recon_loss

            # generator loss
            fake_validity = discriminator(out["recon"])
            g_loss = CONFIG.g_w * ce(fake_validity, torch.ones_like(fake_validity))
            tot_loss['g_loss'] += g_loss.item()
            loss_all = loss_all + g_loss

            # teacher decoder loss
            if  CONFIG.dist and 'teacher_recon' in out and torch.is_tensor(out['teacher_recon']):
                t_recon_loss = CONFIG.t_w * recon_criterion(out["teacher_recon"], img)
                tot_loss['t_recon_loss'] += t_recon_loss.item()
                loss_all = loss_all + t_recon_loss

            # distillation loss
            if  CONFIG.dist and 'dist_loss' in out and torch.is_tensor(out['dist_loss']):
                dist_loss = CONFIG.dist_w * out["dist_loss"]
                tot_loss['dist_loss'] += dist_loss.item()
                loss_all = loss_all + dist_loss

            tot_loss['loss'] += loss_all.item()

            loss_time = time.time()

            opt.zero_grad()
            loss_all.backward()
            summary_grads(model, 'generator', writer, epoch)
            opt.step()

            bw_time = time.time()

            for module in model.modules():
                if isinstance(module, MemoryQueue):
                    module.update()

            del loss_all, recon_loss, g_loss, fake_validity
            if CONFIG.dist:
                del t_recon_loss, dist_loss

            # print('AE time {:.4f}, disc time {:.4f}, disc bw time {:.4f}, loss time {:.4f}, bw time {:.4f}'.format(
            #     ae_time - iter_start, disc_time - ae_time, disc_bw_time - disc_time, loss_time - disc_time, bw_time - loss_time
            # ))
        del out

    # avg loss
    for k, v in tot_loss.items():
        tot_loss[k] /= batches_done

    return tot_loss

def val(dataloader, epoch):
    model.eval()
    tot_loss = {'recon_l1': 0.}

    # for reconstructed img
    reconstructed = []
    # for input img
    inputs = []
    # for anomaly score
    scores = []
    # for gt labels
    labels = []

    count = 0
    for i, (img, label) in enumerate(tqdm(dataloader, disable=CONFIG.disable_tqdm)):
        count += img.shape[0]
        img = img.to(CONFIG.device)
        label = label.to(CONFIG.device)

        opt.zero_grad()

        out = model(img)
        fake_v = discriminator(out['recon'])

        scores += list(fake_v.detach().cpu().numpy())
        labels += list(label.detach().cpu().numpy())
        reconstructed += list(out['recon'].detach().cpu().numpy())
        inputs += list(img.detach().cpu().numpy())

        # this is just an indication
        tot_loss['recon_l1'] += torch.mean(torch.abs(out['recon'] - img)).item()

    tot_loss['recon_l1'] = tot_loss['recon_l1'] / count
    return reconstructed, inputs, scores, labels, tot_loss


def summary_grads(model, model_name, writer, epoch):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(float(p.grad.mean()))
    grads = np.array(grads)
    writer.add_scalar(f'Debug/{model_name} gradient mean', grads.mean(), epoch)
    writer.add_scalar(f'Debug/{model_name} gradient max', grads.max(), epoch)


if __name__ == '__main__':
    main()
