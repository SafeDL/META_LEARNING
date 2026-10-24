"""Joint Bernoulli collision and Gaussian clearance response GP primitives."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from methods.common.classification import torch_moments
from methods.common.gp_classifier import ClassificationKernel


MARGIN_NOISE = .05
MARGIN_CAP_M = 100.


def margin_values(clearance):
    values = np.asarray(clearance,dtype=float)
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError('Joint response models require nonnegative measured clearance')
    return np.log1p(np.minimum(values,MARGIN_CAP_M))/math.log(11)


class JointResponseKernel(ClassificationKernel):
    def __init__(self,dimension,*,uncorrelated=False):
        super().__init__(dimension,deep=True)
        with torch.no_grad():
            self.mean.fill_(-1.4)
        self.margin_mean = nn.Parameter(torch.tensor(.5,dtype=torch.float64))
        self.log_margin_scale = nn.Parameter(torch.tensor(math.log(.3),dtype=torch.float64))
        self.raw_correlation = nn.Parameter(torch.tensor(math.atanh(-.5),dtype=torch.float64),
                                           requires_grad=not uncorrelated)
        self.uncorrelated = uncorrelated

    @property
    def correlation(self):
        return torch.zeros_like(self.raw_correlation) if self.uncorrelated else self.raw_correlation.tanh()

    def joint_mean(self,count):
        return torch.cat((self.mean.expand(count),self.margin_mean.expand(count)))

    def joint_covariance(self,x,other=None):
        base = self.covariance(x,other)
        scale = self.log_margin_scale.exp()
        cross = self.correlation*scale*base
        return torch.cat((torch.cat((base,cross),dim=-1),
                          torch.cat((cross,scale.square()*base),dim=-1)),dim=-2)


def logistic_laplace(mean,covariance,labels):
    labels = torch.as_tensor(labels,dtype=mean.dtype,device=mean.device)
    eye = torch.eye(len(labels),dtype=mean.dtype,device=mean.device)
    covariance = covariance+1e-5*eye
    f = mean
    for _ in range(10):
        p = f.sigmoid()
        weight = (p*(1-p)).clamp_min(1e-9)
        root = weight.sqrt()
        factor = torch.linalg.cholesky(eye+root[..., :,None]*covariance*root[..., None,:])
        b = weight*(f-mean)+labels-p
        rhs = root*(covariance @ b[...,None])[...,0]
        solved = torch.cholesky_solve(rhs[...,None],factor)[...,0]
        alpha = b-root*solved
        f = mean+(covariance @ alpha[...,None])[...,0]
    p = f.sigmoid()
    root = (p*(1-p)).clamp_min(1e-9).sqrt()
    factor = torch.linalg.cholesky(eye+root[..., :,None]*covariance*root[..., None,:])
    alpha = torch.cholesky_solve((f-mean)[...,None],torch.linalg.cholesky(covariance))[...,0]
    loss = (.5*(alpha*(f-mean)).sum(-1)
            + F.binary_cross_entropy_with_logits(f,labels.expand_as(f),reduction='none').sum(-1)
            + factor.diagonal(dim1=-2,dim2=-1).log().sum(-1))
    return loss,alpha,root,factor


def posterior(prior_mean,prior_covariance,indices,labels,margin=None):
    """Condition on paid margins analytically, then on paid collision labels."""
    size = prior_mean.shape[-1]//2
    if prior_mean.shape[-1] != 2*size or prior_covariance.shape[-2:] != (2*size,2*size):
        raise ValueError('Joint latent arrays must contain collision and margin channels')
    selected = torch.as_tensor(indices,dtype=torch.long,device=prior_mean.device)
    if len(selected) != len(labels):
        raise ValueError('Paid scenario indices and collision labels must align')
    mean,covariance = prior_mean,prior_covariance
    loss = prior_mean[...,0]*0
    if not len(selected):
        return loss,mean,covariance
    if margin is not None:
        values = torch.as_tensor(margin,dtype=mean.dtype,device=mean.device)
        if values.ndim != 1 or len(values) != len(selected):
            raise ValueError('Every component must receive the same paid clearance observations')
        margin_indices = size+selected
        cross = covariance[..., :,margin_indices]
        gram = covariance[..., margin_indices,:][..., :,margin_indices]
        eye = torch.eye(len(selected),dtype=mean.dtype,device=mean.device)
        factor = torch.linalg.cholesky(gram+MARGIN_NOISE**2*eye)
        residual = values-mean[...,margin_indices]
        alpha = torch.cholesky_solve(residual[...,None],factor)[...,0]
        loss = (.5*(residual*alpha).sum(-1)
                + factor.diagonal(dim1=-2,dim2=-1).log().sum(-1)
                + .5*len(selected)*math.log(2*math.pi))
        mean = mean+(cross @ alpha[...,None])[...,0]
        solved = torch.linalg.solve_triangular(factor,cross.transpose(-1,-2),upper=False)
        covariance = covariance-solved.transpose(-1,-2) @ solved
    class_mean = mean[...,selected]
    class_covariance = covariance[...,selected,:][..., :,selected]
    binary_loss,alpha,root,factor = logistic_laplace(class_mean,class_covariance,labels)
    cross = covariance[..., :,selected]
    mean = mean+(cross @ alpha[...,None])[...,0]
    solved = torch.linalg.solve_triangular(factor,root[..., :,None]*cross.transpose(-1,-2),upper=False)
    covariance = covariance-solved.transpose(-1,-2) @ solved
    return loss+binary_loss,mean,covariance


def joint_loss(model,x,labels,clearance,*,use_margin=True):
    mean = model.joint_mean(len(x))
    covariance = model.joint_covariance(x)
    values = margin_values(clearance) if use_margin else None
    return posterior(mean,covariance,range(len(x)),labels,values)[0]


def probabilities(mean,covariance):
    size = mean.shape[-1]//2
    return torch_moments(mean[..., :size],covariance.diagonal(dim1=-2,dim2=-1)[..., :size])[0]


def predictive_log_likelihood(mean,covariance,index,collision,clearance,*,use_margin=True):
    """Score the joint new feedback without conditioning on its collision label."""
    if not isinstance(collision,(bool,np.bool_)):
        raise TypeError('Joint feedback requires an actual Boolean collision label')
    size = mean.shape[-1]//2
    class_mean = mean[...,index]
    class_variance = covariance[...,index,index]
    margin_log_likelihood = torch.zeros_like(class_mean)
    if use_margin:
        value = float(margin_values(clearance))
        margin_mean = mean[...,size+index]
        variance = covariance[...,size+index,size+index]+MARGIN_NOISE**2
        residual = value-margin_mean
        cross = covariance[...,index,size+index]
        margin_log_likelihood = -.5*(math.log(2*math.pi)+variance.log()+residual.square()/variance)
        class_mean = class_mean+cross*residual/variance
        class_variance = class_variance-cross.square()/variance
    p = torch_moments(class_mean,class_variance)[0]
    log_label = (p if collision else 1-p).clamp_min(1e-15).log()
    return margin_log_likelihood+log_label,p


def fit_joint(model,x,labels,clearance,*,steps=10,learning_rate=.003,use_margin=True):
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = joint_loss(model,x,labels,clearance,use_margin=use_margin)/len(labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite joint-response GP evidence')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
