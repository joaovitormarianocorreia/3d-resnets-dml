import torch
import resnet
import argparse
import torch.nn as nn

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_depth', default=50, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--n_classes', default=101, type=int, help= 'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_input_channels', default=3, type=int, help='Number of channels on input')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--conv1_t_size', default=7, type=int, help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride', default=1, type=int, help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool', action='store_true', help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_widen_factor', default=1.0, type=float, help='The number of feature maps of resnet is multiplied by this value')
    return parser.parse_args()

if _name_ == '_main_':
    opt = parse_opts()

    # creating model
    model = resnet.generate_model(
        model_depth=opt.model_depth,
        n_classes=1139,
        n_input_channels=opt.n_input_channels,
        shortcut_type=opt.resnet_shortcut,
        conv1_t_size=opt.conv1_t_size,
        conv1_t_stride=opt.conv1_t_stride,
        no_max_pool=opt.no_max_pool,
        widen_factor=opt.resnet_widen_factor)

    # loading pretrained model
    pretrain_path = './data/pretrained/r3d50_KMS_200ep.pth'
    print('loading pretrained model {}'.format(pretrain_path))

    pretrain = torch.load(pretrain_path, map_location='cpu')
    model.load_state_dict(pretrain['state_dict'])
    tmp_model = model
    tmp_model.fc = nn.Linear(tmp_model.fc.in_features, opt.n_classes)
    model = tmp_model