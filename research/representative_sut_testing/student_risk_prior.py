"""Conjugate scale uncertainty for the existing task-mean risk response."""
import math

from .risk_task_prior import TaskMeanRiskGP


# Smallest integer degrees of freedom with a finite fourth moment.
PRIOR_DEGREES_OF_FREEDOM = 5


class StudentTaskRiskGP(TaskMeanRiskGP):
    """Gaussian risk conditional on a shared inverse-gamma covariance scale."""
    def __init__(self, covariance, prior, family, noise):
        super().__init__(covariance, prior, family, noise)
        self.base_noise = noise
        self.predictive_df = PRIOR_DEGREES_OF_FREEDOM
        self.scale_records = []

    def observe(self, index, risk):
        self.noise = self.base_noise
        super().observe(index, risk)
        effective_count = len(self.indices) - self.observed_mean_parameters
        self.predictive_df = PRIOR_DEGREES_OF_FREEDOM + effective_count
        multiplier = (PRIOR_DEGREES_OF_FREEDOM - 2 + self.residual_quadratic) / (self.predictive_df - 2)
        self.variance *= multiplier
        self.noise = self.base_noise * multiplier
        self.scale_records.append({"index": int(index), "observations": len(self.indices),
                                   "mean_parameters": self.observed_mean_parameters,
                                   "effective_count": effective_count, "predictive_df": self.predictive_df,
                                   "residual_quadratic": self.residual_quadratic,
                                   "covariance_multiplier": multiplier})


def add_student_risk_variation(response, templates):
    modes = (templates - templates.mean(axis=1, keepdims=True)) / math.sqrt(templates.shape[1] - 1)
    original = response.gp
    response.gp = StudentTaskRiskGP(original.covariance + modes @ modes.T, original.mean,
                                   response.family, noise=original.noise)
    return response
