import torch
import csv
import resnet
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torchvision import get_image_backend
from torchvision.transforms import transforms
from temporal_transforms import Compose as TemporalCompose
from temporal_transforms import TemporalRandomCrop
from loader import ImageLoaderAccImage, VideoLoader
from videodataset import VideoDataset

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

def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]

def get_fine_tuning_parameters(model):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})
    
    return parameters

def image_name_formatter(x):
    return f'image_{x:05d}.jpg'

def get_training_data(video_path, annotation_path, dataset_name, input_type, file_type, spatial_transform=None, temporal_transform=None, target_transform=None):
    
    if get_image_backend() == 'accimage':
        loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
    else:
        loader = VideoLoader(image_name_formatter)

    video_path_formatter = (
        lambda root_path, label, video_id: root_path / label / video_id)
   
    training_data = VideoDataset(video_path,
                                    annotation_path,
                                    'training',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    video_loader=loader,
                                    video_path_formatter=video_path_formatter)

    return training_data

class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

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
    
    optimizer = SGD(parameters,
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

    val_data, collate_fn = get_validation_data(opt.video_path,
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,
                                               spatial_transform,
                                               temporal_transform)
    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size //
                                                         opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None
