import os
import torch
import _init_paths
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import itertools
import matplotlib
import torchvision.transforms as transforms
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.helper_functions.voc import Voc2007Classification
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
import seaborn as sns
from src.helper_functions.nuswide_asl import NusWideFiltered
from model_and_dataset_loader import parse_model_and_args, load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## LOAD DATASET AND MODEL #############################################

args, model = parse_model_and_args()
data_loader = load_dataset(args)

# # MSCOCO 2014
# # parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
# # parser.add_argument('--model_path', type=str, default='./tresnetm-asl-coco-epoch80')
# # parser.add_argument('--model_name', type=str, default='tresnet_l')
# # parser.add_argument('--num-classes', default=80)
# # parser.add_argument('--dataset_type', type=str, default='MSCOCO 2014')
# # parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# # PASCAL VOC2007
# # parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
# # parser.add_argument('--model-path', default='./tresnetxl-asl-voc-epoch80', type=str)
# # parser.add_argument('--model_name', type=str, default='tresnet_xl')
# # parser.add_argument('--num-classes', default=20)
# # parser.add_argument('--dataset_type', type=str, default='PASCAL VOC2007')
# # parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# # NUS_WIDE
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='..NUS_WIDE')
# parser.add_argument('--model_path', type=str, default='./NUS_WIDE_TRresNet_L_448_65.2.pth')
# parser.add_argument('--model_name', type=str, default='tresnet_l')
# parser.add_argument('--num-classes', default=81)
# parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# #IMPORTANT PARAMETER!
# parser.add_argument('--th', type=float, default=0.5)


# parser.add_argument('-b', '--batch-size', default=1, type=int,
#                     metavar='N', help='mini-batch size (default: 16)')
# parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
#                     help='number of data loading workers (default: 16)')
# args = parse_args(parser)

# ########################## SETUP THE MODEL AND LOAD THE DATA #####################


# # Load the data
# instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
# # data_path_train = args.data
# data_path = '{0}/train2014'.format(args.data)

# ########################## EXPERIMENT LOOP ################

correlations = torch.zeros((args.num_classes, args.num_classes))

# if args.dataset_type == 'MSCOCO 2014':

#     instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
#     data_path = '{0}/train2014'.format(args.data)

#     dataset = CocoDetectionFiltered(data_path,
#                                 instances_path,
#                                 transforms.Compose([
#                                     transforms.Resize((args.image_size, args.image_size)),
#                                     transforms.ToTensor(),
#                                     # normalize, # no need, toTensor does normalization
#                                 ]))

# elif args.dataset_type == 'PASCAL VOC2007':

#     dataset = Voc2007Classification('trainval',
#                                     transform=transforms.Compose([
#                     transforms.Resize((args.image_size, args.image_size)),
#                     transforms.ToTensor(),
#                 ]), train=True)

# elif args.dataset_type == 'NUS_WIDE':

#     dataset = NusWideFiltered('train', transform=transforms.Compose([
#                         transforms.Resize((args.image_size, args.image_size)),
#                         transforms.ToTensor()])
#     )


# # Pytorch Data loader
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=args.batch_size, shuffle=True,
#     num_workers=args.workers, pin_memory=True)

    
for i, (tensor_batch, labels) in enumerate(data_loader):
    
    classes = labels.nonzero()[:,1].tolist()
    pairs = list(itertools.combinations(classes, 2))
    for class1,class2 in pairs:
        correlations[class1,class2] += 1
        correlations[class2,class1] += 1


    

sns.heatmap(correlations)
plt.show()
# plt.savefig("label-correlations-{0}.png".format(args.dataset_type))