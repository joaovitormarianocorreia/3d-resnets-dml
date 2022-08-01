import time
import torch
import resnet
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.transforms import transforms
from temporal_transforms import Compose as TemporalCompose, TemporalRandomCrop

from opts import parse_opts
from stats import AverageMeter, Logger
from classifier import GaussianKernels, find_neighbours
from mixup import update_centres, mixup_data, mixup_criterion
from dataset import get_training_data


def save_model(model, kernel_classifier, centres, save_path, seed):
    torch.save(model.state_dict(), save_path + "/" + seed + "model.pt")
    torch.save(kernel_classifier.state_dict(), save_path + "/" + seed + "classifier.pt")
    torch.save(centres, save_path + "/" + seed + "centres.pt")


if __name__ == '__main__':

    # clean the cache memory before training
    torch.cuda.empty_cache()

    # import options arguments
    opt = parse_opts()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create seed to the run 
    seed = opt.model + "-DS" + opt.dataset + "-EP" + str(opt.n_epochs) + "-SM" + str(opt.scale_mixup) + "-A" + str(opt.alpha) + "-B" + str(opt.beta)

    # creating model
    model = resnet.generate_model(
        model_depth = opt.model_depth,
        n_classes = 51,
        n_input_channels = opt.n_input_channels,
        shortcut_type = opt.resnet_shortcut,
        conv1_t_size = opt.conv1_t_size,
        conv1_t_stride = opt.conv1_t_stride,
        no_max_pool = opt.no_max_pool,
        widen_factor = opt.resnet_widen_factor)

    # load pretrained model
    if opt.load_pretrained:
        print('\nLoading pretrained model\n')
        pretrain = torch.load(opt.pretrained_path, map_location='cpu')
        model.load_state_dict(pretrain, strict=False)

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

    # mixup criterion 
    criterion_mixup = nn.CrossEntropyLoss()

    # create loggers to the session
    train_logger = Logger(opt.result_path / 'train.log', ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(opt.result_path / 'train_batch.log', ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    acc_geral = -1
    best_epoch = -1

    # training loop
    print('\nBegin Training\n')
    for epoch in range(opt.n_epochs):
        
        print(f'\nTrain at epoch {epoch}')

        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        end_time = time.time()

        if (epoch % opt.update_interval) == 0:
            print('\tUpdating kernel centres')
            centres = update_centres(centres, model, update_loader, opt.batch_size, device)
            print('\tFinding training set neighbours')
            centres = centres.cpu()
            neighbours_tr = find_neighbours(opt.num_neighbours, centres )
            centres = centres.to(device)

            if epoch > 0:
                acc_actual = test()
                if (acc_actual >= acc_geral):
                    best_epoch = epoch
                    acc_geral = acc_actual
                    save_model(model, kernel_classifier, centres, opt.save_path, seed)

        running_loss = 0.0
        running_correct = 0
        
        # training epoch
        for i, data in tqdm(enumerate(train_loader, 0)):
            # setting inputs and targets
            inputs, labels, index = data
            inputs = inputs.to(device)
            labels = labels.to(device, non_blocking=True).view(-1)
            index = index.to(device)

            # mixup data
            inputs_mixup, targets_a, targets_b, lam = mixup_data(inputs, labels, device, opt.alpha)
            inputs_mixup, targets_a, targets_b = map(Variable, (inputs_mixup, targets_a, targets_b))
            outputs_mixup = kernel_classifier(model(inputs_mixup), centres, centre_labels, neighbours_tr[index, :])
            loss_mixup = mixup_criterion(criterion_mixup, outputs_mixup, targets_a, targets_b, lam)

            optimizer.zero_grad()

            log_prob = kernel_classifier(model(inputs), centres, centre_labels, neighbours_tr[index,:])
            loss = criterion(log_prob, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pred = log_prob.argmax(dim=1, keepdim=True) 
            correct = pred.eq(labels.view_as(pred)).sum().item()
            running_correct += correct

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            
            train_batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(train_loader) + (i + 1),
                    'loss': losses.val,
                    'acc': accuracies.val,
                    'lr': current_lr
                })

            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))
        

        train_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

        acc = 100. * running_correct / len(train_loader.dataset)
        print(f'| Epoch [{epoch}] | Loss [{loss}] | Accuracy [{acc}] |')

    print("Updating kernel centres (final time).")
    centres = update_centres(centres, model, update_loader, opt.batch_size, device)

    # save model after training
    save_model(model, kernel_classifier, centres, opt.save_path, seed)
