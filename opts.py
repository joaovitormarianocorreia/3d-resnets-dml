import argparse
from pathlib import Path

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='hmdb51', type=str, help='Used dataset (ucf101 | hmdb51)')
    parser.add_argument('--model', default='resnet', type=str, help='Name of the used model used')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--n_classes', default=51, type=int, help= 'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_epochs', default=200, type=int, help= 'Number of epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument('--n_input_channels', default=3, type=int, help='Number of channels on input')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--conv1_t_size', default=7, type=int, help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride', default=1, type=int, help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool', action='store_true', help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_widen_factor', default=1.0, type=float, help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=30, type=int, help='Temporal duration of inputs')
    parser.add_argument('--video_path', default='./data/hmdb/jpg', type=Path, help='Directory path of videos')
    parser.add_argument('--annotation_path', default='./data/hmdb/json/hmdb51_1.json', type=Path, help='Annotation file path')
    parser.add_argument('--ft_begin_module', default='',type=str, help='Module name of beginning of fine-tuning (conv1, layer1, fc, denseblock1, classifier, ...). The default means all layers are fine-tuned.')
    parser.add_argument('--load_pretrained', default=False, type=bool, help='If true load pretrained model')
    parser.add_argument('--pretrained_path', default='', type=str, help='Path to the pretrained model')
    parser.add_argument('--save_path', default='./data/state/', type=str, help='Path of the save models dir')
    parser.add_argument('--result_path', default=None, type=Path, help='Result directory path')
    # gaussian kernel classifier
    parser.add_argument('--sigma', default=10, type=int, help='Gaussian sigma.')
    parser.add_argument('--num_neighbours', default=200, type=int, help='Number of Gaussian Kernel Classifier neighbours.')
    parser.add_argument('--update_interval', default=5, type=int, help='Stored centres/neighbours are updated every update_interval epochs.')
    # mixup
    parser.add_argument('--alpha', default=1, type=float,help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--scale_mixup', default=0.0001, type=float,help='scaling the mixup loss')
    parser.add_argument('--beta', default=1, type=float,help='scaling the gauss loss')

    return parser.parse_args()