import argparse
from model.discriminator import Discriminator
from model.decoder import Decoder
from model.encoder import Encoder
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import CelebADataset
from lib.criterion import ReconstructionCriterion, KLCriterion, ClassificationCriterion
from torch.utils.data import DataLoader
import os
import torch
from os import path
import time
import shutil
from random import sample
from torch.utils.tensorboard import SummaryWriter
import pickle
import ast
from torch import nn
from torchvision import utils


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='Pytorch Training DCGAN for CelebA Dataset')
parser.add_argument('-bp', '--base_path', default="/data/fhz")
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# dataset parameters
parser.add_argument('-nl', '--need-label', action='store_true', help='if we need the label for each image')
parser.add_argument('-is', '--image-size', default=64, type=int, help="The crop image size")
# optimizer parameters
parser.add_argument('--lrd', '--learning-rate-d-d', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate for discriminator and decoder')
parser.add_argument('--lre', '--learning-rate-encoder', default=1e-8, type=float,
                    metavar='LR', help='initial learning rate for encoder')
parser.add_argument('-b1', '--beta1', default=0.5, type=float, metavar='Beta1 In ADAM', help='beta1 for adam')
# network parameters
parser.add_argument('-nc', '--num-channel', default=3, type=int, help="The image channel")
parser.add_argument('-ld', '--latent-dim', default=100, type=int, help="The latent dim for generator")
parser.add_argument('--vnf', '--vae-num-feature', default=64, type=int, help="The feature number for VAE part")
parser.add_argument('--dnf', '--discriminator-num-feature', default=64, type=int,
                    help="The feature number for discriminator")
parser.add_argument('--gr', '--gamma-for-reconstruction', default=1e-6, type=int,
                    help="The gamma for the reconstruction loss")
# criterion parameters
parser.add_argument('--wpe', '--warm-up-epoch', default=5, type=float,
                    help="The warm up epoch for kl divergency part in encoder")
parser.add_argument("-s", "--sigma", default=1, type=float,
                    help="The standard variance for reconstructed features, work as regularization")
# real and fake image label
parser.add_argument('-rl', '--real-label', default=1, type=int, help="The label for real image in discriminator")
parser.add_argument('-fl', '--fake-label', default=0, type=int, help="The label for fake image in discriminator")
# Some maybe useful parameters
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
# GPU and data parallel parameters
parser.add_argument("--gpu", default="0,1", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
parser.add_argument('-dp', '--data-parallel', action='store_true', help='Use Data Parallel')


def main():
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataset_base_path = path.join(args.base_path, "dataset", "celeba")
    image_base_path = path.join(dataset_base_path, "img_align_celeba")
    split_dataset_path = path.join(dataset_base_path, "Eval", "list_eval_partition.txt")
    with open(split_dataset_path, "r") as f:
        split_annotation = f.read().splitlines()
    # create the data name list for train,test and valid
    train_data_name_list = []
    test_data_name_list = []
    valid_data_name_list = []
    for item in split_annotation:
        item = item.split(" ")
        if item[1] == '0':
            train_data_name_list.append(item[0])
        elif item[1] == '1':
            valid_data_name_list.append(item[0])
        else:
            test_data_name_list.append(item[0])
    attribute_annotation_dict = None
    if args.need_label:
        attribute_annotation_path = path.join(dataset_base_path, "Anno", "list_attr_celeba.txt")
        with open(attribute_annotation_path, "r") as f:
            attribute_annotation = f.read().splitlines()
        attribute_annotation = attribute_annotation[2:]
        attribute_annotation_dict = {}
        for item in attribute_annotation:
            img_name, attribute = item.split(" ", 1)
            attribute = tuple([eval(attr) for attr in attribute.split(" ") if attr != ""])
            assert len(attribute) == 40, "the attribute of item {} is not equal to 40".format(img_name)
            attribute_annotation_dict[img_name] = attribute
    discriminator = Discriminator(num_channel=args.num_channel, num_feature=args.dnf,
                                  data_parallel=args.data_parallel).cuda()
    decoder = Decoder(latent_dim=args.latent_dim, num_feature=args.vnf,
                      num_channel=args.num_channel, data_parallel=args.data_parallel).cuda()
    encoder = Encoder(num_channel=args.num_channel, num_feature=args.dnf, latent_dim=args.latent_dim,
                      data_parallel=args.data_parallel).cuda()
    input("Begin the {} time's training, the train dataset has {} images and the valid has {} images".format(
        args.train_time, len(train_data_name_list), len(valid_data_name_list)))
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lrd, betas=(args.beta1, 0.999))
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lrd, betas=(args.beta1, 0.999))
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lre, betas=(args.beta1, 0.999))
    writer_log_dir = "{}/VAEGAN/runs/train_time:{}".format(args.base_path, args.train_time)
    # Here we implement the resume part
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.not_resume_arg:
                args = checkpoint['args']
                args.start_epoch = checkpoint['epoch']
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            decoder.load_state_dict(checkpoint["decoder_state_dict"])
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            dis_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            dec_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            enc_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    else:
        if os.path.exists(writer_log_dir):
            flag = input("DCGAN train_time:{} will be removed, input yes to continue:".format(
                args.train_time))
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    # Here we just use the train dset in training
    train_dset = CelebADataset(base_path=image_base_path, data_name_list=train_data_name_list,
                               image_size=args.image_size,
                               label_dict=attribute_annotation_dict)
    train_dloader = DataLoader(dataset=train_dset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    valid_dset = CelebADataset(base_path=image_base_path, data_name_list=valid_data_name_list,
                               image_size=args.image_size,
                               label_dict=attribute_annotation_dict)
    valid_dloader = DataLoader(dataset=valid_dset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    recon_criterion = ReconstructionCriterion(sigma=args.sigma)
    cls_criterion = ClassificationCriterion()
    kl_criterion = KLCriterion()
    for epoch in range(args.start_epoch, args.epochs):
        train(train_dloader, valid_dloader, encoder, decoder, discriminator, enc_optimizer, dec_optimizer,
              dis_optimizer, recon_criterion, cls_criterion, kl_criterion, writer, epoch)
        # save parameters
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            "discriminator_state_dict": discriminator.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            'discriminator_optimizer': dis_optimizer.state_dict(),
            'decoder_optimizer': dec_optimizer.state_dict(),
            'encoder_optimizer': enc_optimizer.state_dict()
        })


def train(train_dloader, valid_dloader, encoder, decoder, discriminator, enc_optimizer, dec_optimizer, dis_optimizer,
          recon_criterion, cls_criterion, kl_criterion, writer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    reconstruction_loss = AverageMeter()
    KL_loss = AverageMeter()
    discriminator_real_loss = AverageMeter()
    discriminator_fake_loss = AverageMeter()
    decoder_fake_loss = AverageMeter()
    encoder.train()
    decoder.train()
    discriminator.train()
    end = time.time()
    """
    Train strategy:
    Train discriminator:
        1. Calculate the real image loss
        2. Calculate the fake image generated from encoder's inference's loss
        3. Calculate the fake image generated from random noise's loss
    Train decoder:
        1. Confuse discriminator loss
        2. Calculate the feature reconstruction loss with weight in args
    Train encoder
        3. Train with the ELBO loss
    """
    for i, (real_image, index, *_) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        real_image = real_image.cuda()
        batch_size = real_image.size(0)
        # basic calculation
        mu, log_sigma, sigma, z_sample = encoder(real_image)
        # here we train discriminator
        dis_optimizer.zero_grad()
        # discriminate real image
        real_label = torch.full((batch_size,), args.real_label).cuda()
        output, *_ = discriminator(real_image).view(-1)
        dis_x = output.mean().item()
        dis_loss_real = cls_criterion(output, real_label)
        dis_loss_real.backward()
        # discriminate fake image from inference
        fake_inf = decoder(z_sample)
        fake_inf_label = torch.full((batch_size,), args.fake_label).cuda()
        output, *_ = discriminator(fake_inf.detach()).view(-1)
        dis_gz_inf = output.mean().item()
        dis_loss_fake_inf = 0.5 * cls_criterion(output, fake_inf_label)
        dis_loss_fake_inf.backward()
        # discriminate fake image from random generation
        noise = torch.randn(batch_size, args.latent_dim, 1, 1).cuda()
        fake_gen = decoder(noise)
        fake_gen_label = torch.full((batch_size,), args.fake_label).cuda()
        output, *_ = discriminator(fake_gen.detach()).view(-1)
        dis_gz_gen = output.mean().item()
        dis_loss_fake_gen = 0.5 * cls_criterion(output, fake_gen_label)
        dis_loss_fake_gen.backward()
        # optimize discriminator
        dis_optimizer.step()

        # here we train generator to make their generator image looks more real
        dec_optimizer.zero_grad()
        dec_label_inf = torch.full((batch_size,), args.real_label).cuda()
        output, features_inf = discriminator(fake_inf).view(-1)
        dis_gz_inf_2 = output.mean().item()
        dec_loss_inf = 0.5 * cls_criterion(output, dec_label_inf)
        dec_loss_inf.backward()
        dec_label_gen = torch.full((batch_size,), args.real_label).cuda()
        output, *_ = discriminator(fake_gen).view(-1)
        dis_gz_gen_2 = output.mean().item()
        dec_loss_gen = 0.5 * cls_criterion(output, dec_label_gen)
        dec_loss_gen.backward()
        _, features_raw = discriminator(real_image)
        recon_loss = recon_criterion(features_raw, features_inf)
        recon_loss_dec = args.gr * recon_loss
        recon_loss_dec.backward(retain_graph=True)
        dec_optimizer.step()

        # here we train the encoder with kl divergence
        enc_optimizer.zero_grad()
        kl_loss = kl_criterion(mu, log_sigma, sigma)
        kl_loss_beta = (epoch + 1) / args.wpe
        enc_loss = kl_loss * kl_loss_beta + recon_loss
        enc_loss.backward()
        enc_optimizer.step()
        """
        The loss needs to be record are:
        1. feature reconstruction loss
        2. kl divergency loss
        3. discriminator loss for fake image with discriminator training
        4. discriminator loss for real image with discriminator training
        5. discriminator loss for fake image with decoder training
        """
        reconstruction_loss.update(float(recon_loss))
        KL_loss.update(float(kl_loss))
        discriminator_fake_loss.update(float(dis_loss_fake_gen + dis_loss_fake_inf))
        discriminator_real_loss.update(float(dis_loss_real))
        decoder_fake_loss.update(float(dec_loss_gen + dec_loss_inf))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'D(x) [{3:.4f}] D(G(z))(inf) [{4:.4f}/{5:.4f}] D(G(z))(gen) [{6:.4f}/{7:.4f}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Discriminator Real Loss {disrl.val:.4f} ({disrl.avg:.4f})\t' \
                         'Discriminator Fake Loss {disfl.val:.4f} ({disfl.avg:.4f})\t' \
                         'Decoder Fake Loss {decfl.val:.4f} ({decfl.avg:.4f})\t' \
                         'KL Loss {kl.val:.4f} ({kl.avg:.4f})\t' \
                         'Reconstruction Loss {recl.val:.4f} ({recl.avg:.4f})\t'.format(
                epoch, i + 1, len(train_dloader), dis_x, dis_gz_inf, dis_gz_inf_2, dis_gz_gen, dis_gz_gen_2,
                batch_time=batch_time,
                data_time=data_time, disrl=discriminator_real_loss, disfl=discriminator_fake_loss,
                decfl=decoder_fake_loss,
                kl=KL_loss, recl=reconstruction_loss)
            print(train_text)
    writer.add_scalar(tag="Discriminator Real Loss", scalar_value=discriminator_real_loss.avg, global_step=epoch)
    writer.add_scalar(tag="Discriminator Fake Loss", scalar_value=discriminator_fake_loss.avg, global_step=epoch)
    writer.add_scalar(tag="Decoder Fake Loss", scalar_value=decoder_fake_loss.avg, global_step=epoch)
    writer.add_scalar(tag="KL Loss", scalar_value=KL_loss.avg, global_step=epoch)
    writer.add_scalar(tag="Reconstruction Loss", scalar_value=reconstruction_loss.avg, global_step=epoch)
    # in the train end, we want to test some real images and fake images from the train dataset and the valid dataset
    # train dataset
    real_image = utils.make_grid(real_image[:64, ...], nrow=8)
    writer.add_image(tag="Train/Real_Image", img_tensor=(real_image * 0.5) + 0.5, global_step=epoch)
    fake_gen = utils.make_grid(fake_gen[:64, ...].detach(), nrow=8)
    writer.add_image(tag="Train/Fake_Image_Gen", img_tensor=(fake_gen * 0.5) + 0.5, global_step=epoch)
    fake_inf = utils.make_grid(fake_inf[:64, ...].detach(), nrow=8)
    writer.add_image(tag="Train/Fake_Image_Inf", img_tensor=(fake_inf * 0.5) + 0.5, global_step=epoch)
    # valid dataset
    valid_dloader = enumerate(valid_dloader)
    i, (real_image, index, *_) = valid_dloader.__next__()
    with torch.no_grad():
        mu, log_sigma, sigma, z_sample = encoder(real_image)
        fake_inf = decoder(z_sample)
        noise = torch.randn(batch_size, args.latent_dim, 1, 1).cuda()
        fake_gen = decoder(noise)
        writer.add_image(tag="Valid/Real_Image", img_tensor=(real_image * 0.5) + 0.5, global_step=epoch)
        writer.add_image(tag="Valid/Fake_Image_Gen", img_tensor=(fake_gen * 0.5) + 0.5, global_step=epoch)
        writer.add_image(tag="Valid/Fake_Image_Inf", img_tensor=(fake_inf * 0.5) + 0.5, global_step=epoch)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filefolder = '{}/VAEGAN/parameter/train_time_{}'.format(args.base_path, state["args"].train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()
