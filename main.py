import torch
import csv
import resnet
import time
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torchvision import get_image_backend
from torchvision.transforms import transforms
from temporal_transforms import Compose as TemporalCompose
from temporal_transforms import TemporalRandomCrop
from loader import ImageLoaderAccImage, VideoLoader
from videodataset import VideoDataset, VideoDatasetMultiClips
from .utils import save_checkpoint, get_lr, calculate_accuracy, get_fine_tuning_parameters, image_name_formatter
from .utils import Logger, AverageMeter

# Opts 
model_depth = 50 # Depth of the model
n_classes = 51 # Number of classes of the Dataset
n_input_channels = 3 # Number of channels of the input dataset
resnet_shortcut = 'B' # Type of the ResNet shortcut
conv1_t_size = 7 # Kernel size in t dim of conv1
conv1_t_stride = 1 # Stride in t dim of conv1
no_max_pool = False # If True, the max pool after conv1 is removed
resnet_widen_factor = 1.0 # The number of feature maps of resnet is multiplied by this value
pretrain_path = '' # Path to pretrained model (.pth)
n_finetune_classes = n_classes
ft_begin_module = '' # Module name of beginning of fine-tuning (conv1, layer1, fc, denseblock1...). If default all layers are fine tuned
sample_duration = 16 # Duration of the samples
sample_size = 112 # Height and width of inputs
video_path = '' # Path to the videos dir
annotation_path = '' # Path to the annotation file
result_path = '' # Result directory path
dataset = '' # Used dataset
input_type = 'rgb'
file_type = 'jpg'
batch_size = 128
learning_rate = 0.01
momentum = 0.9
weight_decay = 1e-3
multistep_milestones = [50, 100, 150] # Milestones of LR scheduler. See documentation of MultistepLR
resume_path = '' # Save data (.pth) of previous training
n_val_samples = 3 # Number of validation samples for each activity
n_epochs = 200 # Number of total epochs to run

def generate_model():
    model = resnet.generate_model(
        model_depth=model_depth,
        n_classes=n_classes,
        n_input_channels=n_input_channels,
        shortcut_type=resnet_shortcut,
        conv1_t_size=conv1_t_size,
        conv1_t_stride=conv1_t_stride,
        no_max_pool=no_max_pool,
        widen_factor=resnet_widen_factor)
    return model

def load_pretrained_model(model):
    if True:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        tmp_model.fc = nn.Linear(tmp_model.fc.in_features, n_finetune_classes)

    return model



def get_training_data(video_path, annotation_path, dataset_name, input_type, file_type, spatial_transform=None, temporal_transform=None, target_transform=None):
    
    if get_image_backend() == 'accimage':
        loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
    else:
        loader = VideoLoader(image_name_formatter)

    video_path_formatter = (
        lambda root_path, label, video_id: root_path / label / video_id)
   
    training_data = VideoDataset(
        video_path,
        annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter)

    return training_data


def get_validation_data(video_path, annotation_path, dataset_name, input_type, file_type, spatial_transform=None, temporal_transform=None, target_transform=None):
    assert input_type in ['rgb']
    assert file_type in ['jpg']

    if get_image_backend() == 'accimage':
        loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
    else:
        loader = VideoLoader(image_name_formatter)

    video_path_formatter = (lambda root_path, label, video_id: root_path / label / video_id)
    validation_data = VideoDatasetMultiClips(
        video_path,
        annotation_path,
        'validation',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter)

    return validation_data, collate_fn



def train_epoch(epoch, data_loader, model, criterion, optimizer, device, current_lr, epoch_logger, batch_logger, tb_writer=None):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': current_lr
            })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses,
                                                         acc=accuracies))

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', accuracies.avg, epoch)

def val_epoch(epoch, data_loader, model, criterion, device, logger, tb_writer=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)

    return losses.avg

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Generate model
    model = generate_model()
    # Load pretrained
    model = load_pretrained_model(model)
    # Load fine tuning parameters
    parameters = get_fine_tuning_parameters(model)
    # Define loss function
    criterion = CrossEntropyLoss().to(device)

    # Set up spatial transforms to data
    spatial_transform = []
    spatial_transform.append(transforms.Resize(sample_size))
    spatial_transform.append(transforms.CenterCrop(sample_size))
    spatial_transform.append(transforms.RandomHorizontalFlip())
    spatial_transform.append(transforms.ColorJitter())
    spatial_transform.append(transforms.ToTensor())
    spatial_transform.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    spatial_transform = transforms.Compose(spatial_transform)

    # Set up temporal transforms to data
    temporal_transform = []
    temporal_transform.append(TemporalRandomCrop(sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    # Get training data
    train_data = get_training_data(video_path, annotation_path, dataset, input_type, file_type, spatial_transform, temporal_transform)
    train_sampler = None
    
    # Set train loader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        sampler=train_sampler)

    # Set train logger and batch logger
    train_logger = Logger(result_path / 'train.log', ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(result_path / 'train_batch.log', ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    
    dampening = 0.0
    
    optimizer = SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, multistep_milestones)

    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scheduler.milestones = multistep_milestones

    # Set validation
    spatial_transform = [
        transforms.Resize(sample_size),
        transforms.CenterCrop(sample_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    spatial_transform = transforms.Compose(spatial_transform)

    temporal_transform = [TemporalRandomCrop(sample_duration)]
    temporal_transform = TemporalCompose(temporal_transform)

    val_data, collate_fn = get_validation_data(video_path, annotation_path, dataset, input_type, file_type, spatial_transform, temporal_transform)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=(batch_size // n_val_samples),
        shuffle=False,
        pin_memory=True,
        sampler=val_sampler)

    val_logger = None

    tb_writer = None

    # Training loop
    prev_val_loss = None 
    for i in range(begin_epoch, n_epochs + 1):
        current_lr = get_lr(optimizer)
        train_epoch(i, train_loader, model, criterion, optimizer, device, current_lr, train_logger, train_batch_logger, tb_writer)

        if i % 10 == 0:
            save_file_path = result_path / 'save_{}.pth'.format(i)
            save_checkpoint(save_file_path, i, checkpoint['arch'], model, optimizer, scheduler)

        prev_val_loss = val_epoch(i, val_loader, model, criterion, device, val_logger, tb_writer)

        scheduler.step()
