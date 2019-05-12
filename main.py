import argparse
from model.discriminator import Discriminator
from model.decoder import Decoder
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import CelebADataset
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
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b1', '--beta1', default=0.5, type=float, metavar='Beta1 In ADAM', help='beta1 for adam')
# network parameters
parser.add_argument('-nc', '--num-channel', default=3, type=int, help="The image channel")
parser.add_argument('-ld', '--latent-dim', default=100, type=int, help="The latent dim for generator")
parser.add_argument('--gnf', '--generator-num-feature', default=64, type=int, help="The feature number for generator")
parser.add_argument('--dnf', '--discriminator-num-feature', default=64, type=int,
                    help="The feature number for discriminator")
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
    generator = Decoder(latent_dim=args.latent_dim, num_feature=args.gnf,
                        num_channel=args.num_channel, data_parallel=args.data_parallel).cuda()
    input("Begin the {} time's training, the train dataset has {} images and the valid has {} images".format(
        args.train_time, len(train_data_name_list), len(valid_data_name_list)))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    writer_log_dir = "{}/DCGAN/runs/train_time:{}".format(args.base_path, args.train_time)
    # Here we implement the resume part
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.not_resume_arg:
                args = checkpoint['args']
                args.start_epoch = checkpoint['epoch']
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            generator.load_state_dict(checkpoint["generator_state_dict"])
            d_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            g_optimizer.load_state_dict(checkpoint['generator_optimizer'])
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
    criterion = nn.BCELoss()
    for epoch in range(args.start_epoch, args.epochs):
        train(train_dloader, generator, discriminator, g_optimizer, d_optimizer, criterion, writer, epoch)
        # save parameters
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            "discriminator_state_dict": discriminator.state_dict(),
            "generator_state_dict": generator.state_dict(),
            'discriminator_optimizer': d_optimizer.state_dict(),
            'generator_optimizer': g_optimizer.state_dict()
        })


def train(train_dloader, generator, discriminator, g_optimizer, d_optimizer, criterion, writer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    generator_loss = AverageMeter()
    discriminator_real_loss = AverageMeter()
    discriminator_fake_loss = AverageMeter()
    generator.train()
    discriminator.train()
    end = time.time()
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()
    for i, (real_image, index, *_) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        real_image = real_image.cuda()
        batch_size = real_image.size(0)
        # create noise for generator
        noise = torch.randn(batch_size, args.latent_dim, 1, 1).cuda()
        # create image label
        real_label = torch.full((batch_size,), args.real_label).cuda()
        # use discriminator to distinguish the real images
        output = discriminator(real_image).view(-1)
        # calculate d_x
        d_x = output.mean().item()
        # calculate the discriminator loss in real image
        d_loss_real = criterion(output, real_label)
        # use discriminator to distinguish the fake images
        fake = generator(noise)
        # here we only train discriminator, so we use fake.detach()
        output = discriminator(fake.detach()).view(-1)
        # calculate d_gz_1
        d_gz_1 = output.mean().item()
        # create fake label
        fake_label = torch.full((batch_size,), args.fake_label).cuda()
        # calculate the discriminator loss in fake image
        d_loss_fake = criterion(output, fake_label)
        # optimize discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        # here we train generator to make their generator image looks more real
        # one trick for generator is to use  max log(D) instead of min log(1-D)
        g_label = torch.full((batch_size,), args.real_label).cuda()
        output = discriminator(fake).view(-1)
        # calculate d_gz_2
        d_gz_2 = output.mean().item()
        # calculate the g_loss
        g_loss = criterion(output, g_label)
        g_loss.backward()
        g_optimizer.step()
        # zero grad each optimizer
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        # update d loss and g loss
        generator_loss.update(float(g_loss))
        discriminator_fake_loss.update(float(d_loss_fake))
        discriminator_real_loss.update(float(d_loss_real))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'D(x) [{3:.4f}] D(G(z)) [{4:.4f}/{5:.4f}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Discriminator Real Loss {drl.val:.4f} ({drl.avg:.4f})\t' \
                         'Discriminator Fake Loss {dfl.val:.4f} ({dfl.avg:.4f})\t' \
                         'Generator Loss {gl.val:.4f} ({gl.avg:.4f})\t'.format(
                epoch, i + 1, len(train_dloader), d_x, d_gz_1, d_gz_2, batch_time=batch_time,
                data_time=data_time, drl=discriminator_real_loss, dfl=discriminator_fake_loss, gl=generator_loss)
            print(train_text)
    writer.add_scalar(tag="DCGAN/DRL", scalar_value=discriminator_real_loss.avg, global_step=epoch)
    writer.add_scalar(tag="DCGAN/DFL", scalar_value=discriminator_fake_loss.avg, global_step=epoch)
    writer.add_scalar(tag="DCGAN/GL", scalar_value=generator_loss.avg, global_step=epoch)
    # in the train end, we want to add some real images and fake images
    real_image = utils.make_grid(real_image[:64, ...], nrow=8)
    writer.add_image(tag="Real_Image", img_tensor=(real_image * 0.5) + 0.5, global_step=epoch)
    noise = torch.randn(64, args.latent_dim, 1, 1).cuda()
    fake_image = utils.make_grid(generator(noise), nrow=8)
    writer.add_image(tag="Fake_Image", img_tensor=(fake_image * 0.5) + 0.5, global_step=epoch)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filefolder = '{}/DGCAN/parameter/train_time_{}'.format(args.base_path, state["args"].train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()
