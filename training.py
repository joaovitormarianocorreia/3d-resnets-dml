from urllib import response
from sklearn.utils import shuffle
import torch
import argparse
import resnet
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from pathlib import Path
from torchvision.transforms import transforms
from temporal_transforms import Compose as TemporalCompose, TemporalRandomCrop
from loader import VideoLoader
from videodataset import VideoDataset
from model import get_fine_tuning_parameters
from classifier import GaussianKernels, find_neighbours

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
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
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
    parser.add_argument('--ft_begin_module', default='',type=str, help=('Module name of beginning of fine-tuning (conv1, layer1, fc, denseblock1, classifier, ...). The default means all layers are fine-tuned.'))
    # gaussian kernel classifier
    parser.add_argument('--sigma', default=10, type=int, help='Gaussian sigma.')
    parser.add_argument('--num_neighbours', default=200, type=int, help='Number of Gaussian Kernel Classifier neighbours.')
    parser.add_argument('--update_interval', default=5, type=int, help='Stored centres/neighbours are updated every update_interval epochs.')


    return parser.parse_args()

def update_centres(centres, model, update_loader, batch_size):

	# disable dropout, use global stats for batchnorm
	model.eval()

	# disable learning
	with torch.no_grad():

		# update stored centres
		for i, data in enumerate(update_loader, 0):

			# get the inputs; data is a list of [inputs, labels]. Send to GPU
			inputs, labels, index = data
			inputs = inputs.to(device)

			# extract features for batch
			extracted_features = model(inputs)

			# save to centres tensor
			idx = i*batch_size
			centres[idx:idx + extracted_features.shape[0], :] = extracted_features

	#model.train()
	model.eval()

	return centres



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
    print('Loading pretrained model\n')
    pretrain_path = './data/pretrained/r3d50_KMS_200ep.pth'
    pretrain = torch.load(pretrain_path, map_location='cpu')
    model.load_state_dict(pretrain['state_dict'])

    modules = list(model.children())[:-1]
    modules.append(nn.Flatten())
    model = nn.Sequential(*modules)
    model = model.to(device)

    # parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    
    # set up spatial transforms to data
    spatial_transform = transforms.Compose([
        transforms.Resize(opt.sample_size),
        transforms.CenterCrop(opt.sample_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    update_transform = transforms.Compose([
        transforms.Resize(opt.sample_size),
        transforms.CenterCrop(opt.sample_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # set up temporal transforms to data
    temporal_transform = TemporalCompose([TemporalRandomCrop(opt.sample_duration)])

    print('\nLoading data')

    # get training data
    train_data = get_training_data(opt.video_path, opt.annotation_path, spatial_transform, temporal_transform)

    update_data = get_training_data(opt.video_path, opt.annotation_path, update_transform, temporal_transform)

    # set train loader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = opt.batch_size,
        shuffle = True
    )

    update_loader = torch.utils.data.DataLoader(
        update_data, 
        batch_size = opt.batch_size,
        shuffle = False
    )

    num_train = len(update_loader.dataset)
    print(f'\nNum Train: {num_train}')

    with torch.no_grad():
        num_dims = model(torch.randn(16, 3, 16, opt.sample_size, opt.sample_size).to(device)).size(1)
    print(f'\nNum Dims: {num_dims}')

    # create tensor to store kernel centres 
    centres = torch.zeros(num_train, num_dims).type(torch.FloatTensor).to(device)
    print(f'\nSize of centres: {centres.size()}')

    # create tensor to store labels of centres
    centre_labels = torch.LongTensor(update_data.get_all_labels()).to(device)
    print(f'\nCentre labels: {centre_labels}')

    # create Gaussian Kernel Classifier
    kernel_classifier = GaussianKernels(opt.n_classes, opt.num_neighbours, num_train, opt.sigma)
    kernel_classifier = kernel_classifier.to(device)

    # set optimizer, loss and scheduler
    optimizer = Adam(
        [
            {'params': model.parameters()},
            {'params': kernel_classifier.parameters(), 'lr':  0.1}
        ], lr = 0.1)

    criterion = nn.NLLLoss().to(device)

    # scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 100, 150]) # set multistep milestones 

    # begin of the training loop
    print('\nBegin Training\n')
    for epoch in range(opt.n_epochs):
        
        print('Train at epoch {}'.format(epoch))

        model.train()

        if (epoch % opt.update_interval) == 0:
            print('\tUpdating kernel centres')
            centres = update_centres(centres, model, update_loader, opt.batch_size)
            print('\tFinding training set neighbours')
            centres = centres.cpu()
            neighbours_tr = find_neighbours(opt.num_neighbours, centres )
            centres = centres.to(device)
            print('\tFinished update')

        running_loss = 0.0
        running_correct = 0

        for i, data in enumerate(train_loader, 0):
            # setting inputs and targets
            inputs, labels, index = data
            inputs = inputs.to(device)
            labels = labels.to(device, non_blocking=True).view(-1)
            index = index.to(device)

            optimizer.zero_grad()

            log_prob = kernel_classifier(model(inputs), centres, centre_labels, neighbours_tr[index,:])
            loss = criterion(log_prob, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pred = log_prob.argmax(dim=1, keepdim=True) 
            correct = pred.eq(labels.view_as(pred)).sum().item()
            running_correct += correct

        acc = 100. * running_correct / len(train_loader.dataset)
        print(f'| Epoch [{epoch}] | Loss [{loss}] | Accuracy [{acc}] |')

    print("Updating kernel centres (final time).")
    centres = update_centres(centres, model, update_loader, opt.batch_size)

