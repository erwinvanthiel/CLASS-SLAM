import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)

class HingeLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(HingeLoss, self).__init__()
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = torch.maximum(0*x,0.5-x)
        negative_loss = torch.maximum(0*x,x-0.5)
        loss = y * positive_loss + (1-y) * negative_loss
        if self.weight is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class LinearLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(LinearLoss, self).__init__()
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = 1-x
        negative_loss = x
        loss = y * positive_loss + (1-y) * negative_loss
        if self.weight is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class MSELoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(MSELoss, self).__init__()
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = (1-x)**2
        negative_loss = x**2
        loss = y * positive_loss + (1-y) * negative_loss
        if self.weight is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss


class SLAM(nn.Module):
    
    def __init__(self, coefficients, epsilon, max_eps, num_classes, q=0.5, weight=None, size_average=True):
        super(SLAM, self).__init__()
        
        if epsilon >= max_eps:
            self.p = 1
        else:
            estimate = np.maximum(0, poly.polyval(epsilon, coefficients))
            self.p = np.minimum(1,(estimate / num_classes))

        self.q = q
        self.weight = weight

    def forward(self, x, y):
        bce = torch.nn.BCELoss(weight=self.weight)
        log_loss = bce(x,y)
        linear_loss = (y * (1-x) + (1-y) * x)

        if self.weight is not None:
            loss = loss * self.weight
        loss_total = (1 - self.q*self.p) * torch.mean(linear_loss) + self.q*self.p * log_loss
        return loss_total
