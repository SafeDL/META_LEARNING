import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp, stdtr
from scipy.stats import multivariate_t, t

from methods.historical_expert_mixture.config import JITTER, NOISE
from methods.historical_expert_mixture.posterior import EvidenceMixture
from methods.scale_adaptive_expert_mixture.posterior import PRIOR_DF, ScaleMixture, positive_student_part


def test_sequential_student_evidence_matches_joint_marginal_likelihood():
    covariance = np.array([[.3,.1,.05],[.1,.25,.02],[.05,.02,.2]])
    prior = np.array([[.1,.8],[.2,.7],[.4,.3]])
    probability = np.array([.4,.6])
    model = ScaleMixture(covariance,prior,probability)
    indices,values = [2,0],[.3,.9]
    for i,r in zip(indices,values):
        model.observe(i,r)
    shape = (covariance[np.ix_(indices,indices)]+(NOISE+JITTER)*np.eye(2))*(PRIOR_DF-2)/PRIOR_DF
    evidence = np.array([multivariate_t.logpdf(values,prior[indices,j],shape,df=PRIOR_DF) for j in range(2)])
    expected = np.log(probability)+evidence
    np.testing.assert_allclose(model.log_probability,expected-logsumexp(expected),atol=1e-12)
    np.testing.assert_allclose(sum(r['mixture_log_density'] for r in model.evidence),logsumexp(expected),atol=1e-12)


def test_residual_conditioning_matches_gaussian_but_scales_follow_errors():
    covariance = np.eye(3)*.02+.01
    priors = np.array([[.9,.1],[.8,.2],[.6,.4]])
    model = ScaleMixture(covariance,priors,[.5,.5])
    gaussian = EvidenceMixture(covariance,priors,[.5,.5])
    model.observe(0,.9)
    gaussian.observe(0,.9)
    np.testing.assert_allclose(model.mean,gaussian.mean,atol=1e-13)
    assert model.scale_mean[1]>model.scale_mean[0]*3
    assert model.scale_mean[0]<1


def test_expected_clipped_student_risk_matches_density_integral():
    for mean,scale,df in [(.4,.3,5),(.9,.12,10),(-.5,.8,5)]:
        actual=positive_student_part(mean,scale,0,df)-positive_student_part(mean,scale,1,df)
        expected=quad(lambda r:r*t.pdf((r-mean)/scale,df)/scale,0,1,epsabs=1e-12)[0]+1-stdtr(df,(1-mean)/scale)
        np.testing.assert_allclose(actual,expected,atol=1e-11)


def test_initial_covariance_matches_frozen_gaussian_mixture():
    covariance=np.eye(3)*.1
    priors=np.array([[.8,.2],[.7,.3],[.1,.9]])
    student=ScaleMixture(covariance,priors,[.5,.5])
    gaussian=EvidenceMixture(covariance,priors,[.5,.5])
    for a,b in zip(student.predict(),gaussian.predict()):
        np.testing.assert_allclose(a,b,atol=1e-14)
