import cv2
import os
import time
import torch
import numpy as np
import utils
import logging

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch import nn
from collections import defaultdict
from math import log10, sqrt
from options import *
from model.hidden import Hidden
from average_meter import AverageMeter


def PSNR(img1, img2):
    img1, img2 = torch.flatten(img1), torch.flatten(img2)
    mse = torch.dot((img1 - img2), (img1 - img2)) #.item() ** 2
    mse = mse.sum().item()
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse / 12))
    return psnr
    

def avg_bit_acc(msg1, msg2):
    msg1, msg2 = torch.flatten(msg1), torch.flatten(msg2)
    sub = torch.abs(msg1 - msg2)
    for idx in range(sub.size(dim=0)):
	    sub[idx] = 1 if sub[idx] > 0.5 else 0
    lenght = sub.size(dim=0)
    return ((lenght - sub.sum().item()) / lenght) * 100
    
    

def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """

    torch.autograd.set_detect_anomaly(True)
    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
           image = image.to(device)
           message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
           losses, _ = model.train_on_batch([image, message])

           for name, loss in losses.items():
               training_losses[name].update(loss)
           if step % print_each == 0 or step == steps_in_epoch:
               logging.info(
                   'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
               utils.log_progress(training_losses)
               logging.info('-' * 40)
           step += 1
        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
          tb_logger.save_losses(training_losses, epoch)
          tb_logger.save_grads(epoch)
          tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        psnr_sum, avg_bit_sum, ssim_sum, cnt = 0, 0, 0, 0
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message])



            cnt += 1
            psnr_sum += PSNR(image, encoded_images)
            avg_bit_sum += avg_bit_acc(message, decoded_messages)
#            ssim_sum += ssim(image, encoded_images, data_range=255, size_average=False)



            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False
        
        print("psnr: ", psnr_sum / cnt)
        print("avg bit: ", avg_bit_sum / cnt)
#        print("ssim: ", ssim_sum / cnt)
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
