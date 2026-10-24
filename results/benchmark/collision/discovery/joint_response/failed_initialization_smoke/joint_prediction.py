"""Exact marginal extraction from the joint Gaussian/Laplace approximation."""

import torch

from .joint_response import MARGIN_NOISE, logistic_laplace


def marginal_posterior(prior_mean,prior_covariance,indices,labels,margin=None):
    size = prior_mean.shape[-1]//2
    selected = torch.as_tensor(indices,dtype=torch.long,device=prior_mean.device)
    diagonal = prior_covariance.diagonal(dim1=-2,dim2=-1)
    positions = torch.arange(size,device=prior_mean.device)
    cross_response = prior_covariance[...,positions,size+positions]
    if not len(selected):
        return prior_mean,diagonal,cross_response
    mean = prior_mean
    to_class = prior_covariance[..., :,selected]
    if margin is not None:
        values = torch.as_tensor(margin,dtype=mean.dtype,device=mean.device)
        margin_indices = size+selected
        gram = prior_covariance[...,margin_indices,:][..., :,margin_indices]
        eye = torch.eye(len(selected),dtype=mean.dtype,device=mean.device)
        factor = torch.linalg.cholesky(gram+MARGIN_NOISE**2*eye)
        cross = prior_covariance[..., :,margin_indices]
        residual = values-mean[...,margin_indices]
        alpha = torch.cholesky_solve(residual[...,None],factor)[...,0]
        mean = mean+(cross @ alpha[...,None])[...,0]
        solved = torch.linalg.solve_triangular(factor,cross.transpose(-1,-2),upper=False)
        diagonal = diagonal-solved.square().sum(-2)
        cross_response = cross_response-(solved[..., :size]*solved[..., size:]).sum(-2)
        to_class = to_class-solved.transpose(-1,-2) @ solved[..., :,selected]
    _,alpha,root,factor = logistic_laplace(mean[...,selected],to_class[...,selected,:],labels)
    mean = mean+(to_class @ alpha[...,None])[...,0]
    solved = torch.linalg.solve_triangular(factor,root[..., :,None]*to_class.transpose(-1,-2),upper=False)
    diagonal = diagonal-solved.square().sum(-2)
    cross_response = cross_response-(solved[..., :size]*solved[..., size:]).sum(-2)
    return mean,diagonal,cross_response
