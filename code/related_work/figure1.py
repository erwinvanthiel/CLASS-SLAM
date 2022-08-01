import os
import sys
import torch
import _init_paths
from src.helper_functions.helper_functions import parse_args
# from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
# from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from attacks import pgd, fgsm, mi_fgsm, l2_mi_fgm, get_weights_from_correlations
from mlc_attack_losses import SLAM
from sklearn.metrics import auc
from create_q2l_model import create_q2l_model
from model_and_dataset_loader import parse_model_and_args, load_dataset
import numpy.polynomial.polynomial as poly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser()


args, model = parse_model_and_args()
data_loader = load_dataset(args)

flips_list = []
l_infs = []

for i in range(100):

    # Load clean and adv from related work
    clean = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "adv/q2l/MSCOCO_2014/ml_df_clean{0}.npy".format(i)))).cuda()[:1,:,:,:]
    adv = torch.tensor(np.load(os.path.join(os.path.dirname(__file__),"adv/q2l/MSCOCO_2014/ml_df_adv{0}.npy".format(i)))).cuda()[:1,:,:,:]
    
    # VISUALS
    # print(torch.max(clean))
    # print(torch.max(adv))
    # plt.imshow(clean[0].permute(1, 2, 0))
    # plt.show()
    # plt.imshow(adv[0].permute(1, 2, 0))
    # plt.show()

    #clean predictions
    confidences_clean = torch.sigmoid(model(clean.cuda())) 
    pred_clean = (confidences_clean > 0.5).int()

    # adv predictions from related work i.e. mlalp / ml-deepfool / ml-cw
    confidences_adv = torch.sigmoid(model(adv.cuda()))  
    pred_adv_related_work = (torch.sigmoid(confidences_adv) > 0.5).int()

    # attack and predict again, this time our attacks
    coefs = np.load('../experiment_results/{0}-{1}-profile.npy'.format(args.model_type, args.dataset_type))
    epsilons = np.load('../experiment_results/{0}-{1}-profile-epsilons.npy'.format(args.model_type, args.dataset_type))
    epsilon = 0.32
    estimate = int(np.maximum(0, np.minimum(args.num_classes, poly.polyval(epsilon, coefs))))
    # class_weights = get_weights_from_correlations(instance_correlation_matrix, target, output, subset_length, 0, 4, 4)
    # adv = mi_fgsm(model, clean.detach(), 1 - pred_clean, loss_function=SLAM(coefs, epsilon, np.max(epsilons), args.num_classes, q=0.5), eps=epsilon, device="cuda").detach()
    adv = mi_fgsm(model, clean.detach(), 1 - pred_clean, loss_function=torch.nn.BCELoss(), eps=epsilon, device="cuda").detach().cuda()
    pred_adv = (torch.sigmoid(model(adv)) > 0.5).int().cuda()

    flips = torch.sum(torch.logical_xor(pred_clean,pred_adv)).item()
    print(flips)
    l_inf = torch.max(torch.abs(adv - clean)).item()
    l_2 = torch.sqrt(torch.sum((adv - clean) * (adv - clean))).item()

    flips_list.append(flips)
    l_infs.append(l_inf)
    print(i)

print(np.mean(flips_list), np.std(flips_list))
print(np.mean(l_infs), np.std(l_infs))

    # plt.bar([x for x in range(80)],pred1.cpu().numpy()[0,:], color='green')
    # plt.bar([x for x in range(80)],1-pred1.cpu().numpy()[0,:], color='red')
    # plt.show()