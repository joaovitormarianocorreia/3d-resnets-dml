import torch
import argparse
import resnet
import torch.nn as nn
import mixup

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from pathlib import Path
from torchvision.transforms import transforms
from temporal_transforms import Compose as TemporalCompose, TemporalRandomCrop
from loader import VideoLoader
from videodataset import VideoDataset

def image_name_formatter(x):
    return f'image_{x:05d}.jpg'

def get_training_data(video_path, annotation_path, spatial_transform=None, temporal_transform=None, target_transform=None):
    
    loader = VideoLoader(image_name_formatter)

    video_path_formatter = (
        lambda root_path, 
        label, 
        video_id: root_path / label / video_id)
   
    training_data = VideoDataset(
        video_path,
        annotation_path,
        'training',
        spatial_transform = spatial_transform,
        temporal_transform = temporal_transform,
        target_transform = target_transform,
        video_loader = loader,
        video_path_formatter = video_path_formatter)

    return training_data

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_depth', default=50, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--n_classes', default=51, type=int, help= 'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_epochs', default=200, type=int, help= 'Number of epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--n_input_channels', default=3, type=int, help='Number of channels on input')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--conv1_t_size', default=7, type=int, help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride', default=1, type=int, help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool', action='store_true', help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_widen_factor', default=1.0, type=float, help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--video_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument('--annotation_path', default=None, type=Path, help='Annotation file path')
    
    return parser.parse_args()


if __name__ == '__main__':
    # clean the cache memory before training
    torch.cuda.empty_cache()

    # import options arguments
    opt = parse_opts()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # creating model
    model = resnet.generate_model(
        model_depth = opt.model_depth,
        n_classes = 1139,
        n_input_channels = opt.n_input_channels,
        shortcut_type = opt.resnet_shortcut,
        conv1_t_size = opt.conv1_t_size,
        conv1_t_stride = opt.conv1_t_stride,
        no_max_pool = opt.no_max_pool,
        widen_factor = opt.resnet_widen_factor)

    # loading pretrained model
    pretrain_path = './data/pretrained/r3d50_KMS_200ep.pth'
    pretrain = torch.load(pretrain_path, map_location='cpu')
    model.load_state_dict(pretrain['state_dict'])
    model = model.to(device)

    # set up spatial transforms to data
    spatial_transform = transforms.Compose([
        transforms.Resize(opt.sample_size),
        transforms.CenterCrop(opt.sample_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),        
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # set up temporal transforms to data
    temporal_transform = TemporalCompose([TemporalRandomCrop(opt.sample_duration)])

    # get training data
    train_data = get_training_data(opt.video_path, opt.annotation_path, spatial_transform, temporal_transform)
    train_sampler = None

    # set train loader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = opt.batch_size)

	###############

    num_train = len(train_loader.dataset)
    
    with torch.no_grad():
        num_dims = model(torch.randn(1, 3, opt.input_size, opt.input_size).to(device)).size(1)
    
    num_neighbours = 200

	# create tensor to store kernel centres
    centres = torch.zeros(num_train, num_dims).type(torch.FloatTensor).to(device)
    print("Size of centres is {0}".format(centres.size()))

	# create tensor to store labels of centres
    centre_labels = torch.LongTensor(train_data.get_all_labels()).to(device)

	# create gaussian kernel classifier
    kernel_classifier = mixup.GaussianKernels(opt.num_classes, num_neighbours, num_train, opt.sigma)
    kernel_classifier = kernel_classifier.to(device)

    optimiser = Adam([
        {'params': model.parameters()},
        {'params': kernel_classifier.parameters(), 'lr': 0.1}], 
		lr = 0.1)
    
    criterion_mixup = nn.CrossEntropyLoss()

    ###############

    # set optimizer and scheduler
    optimizer = SGD(
        model.parameters(),
        lr = 0.1)

    criterion = CrossEntropyLoss().to(device)

    # set multistep milestones 
    scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 100, 150]) 

    # begin training loop
    for epoch in range(opt.n_epochs):
        
        print('train at epoch {}'.format(epoch))

        model.train()

        if (epoch % opt.update_interval) == 0:
            print("Updating kernel centres...")
            centres = mixup.update_centres()
            print("Finding training set neighbours...")
            centres = centres.cpu()
            neighbours_tr = mixup.find_neighbours( num_neighbours, centres )
            centres = centres.to(device)

        # get the current learning reate
        lrs = []
        for param_group in optimizer.param_groups:
            lr = float(param_group['lr'])
            lrs.append(lr)
        current_lr = max(lrs)

        for i, data in enumerate(train_loader):
            # setting inputs and targets
            inputs, targets = data
            targets = targets.to(device, non_blocking=True)
            
            # mixup
            inputs_mixup, targets_a, targets_b, lam = mixup.mixup_data(inputs, targets, 1, True)
            inputs_mixup, targets_a, targets_b = map(Variable, (inputs_mixup,targets_a, targets_b))
            outputs_mixup, outputs_2 = kernel_classifier( model(inputs_mixup), centres, centre_labels, neighbours_tr[indices, :] )
            loss_mixup = mixup.mixup_criterion(criterion_mixup, outputs_mixup, targets_a, targets_b, lam)
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # calculate accuracy
            with torch.no_grad():
                batch_size = targets.size(0)

                _, pred = outputs.topk(1, 1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1))
                n_correct_elems = correct.float().sum().item()

                acc = n_correct_elems / batch_size

                print(f'Accuracy: {acc}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
