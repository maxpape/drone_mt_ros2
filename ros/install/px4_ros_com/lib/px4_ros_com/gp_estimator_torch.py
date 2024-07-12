#!/usr/bin/env python3

import torch
import gpytorch
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, variance):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )
        self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        self.covar_module.outputscale = torch.tensor(variance)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GP_estimator():
    def __init__(self):
        self.noise_variance_lin = 0.25
        self.length_hypers = [[1, 5, 7, 7, 7, 7]] * 6
        self.scale_hypers = [1] * 6

        self.models = []
        self.likelihoods = []
        for i in range(6):
            train_x = torch.ones((1, 6))
            train_y = torch.ones((1, 1))
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = torch.tensor(self.noise_variance_lin)
            model = ExactGPModel(train_x, train_y.flatten(), likelihood, self.length_hypers[i], self.scale_hypers[i])
            self.models.append(model)
            self.likelihoods.append(likelihood)

        self.online_regression = False
        self.show_lin = True

    def predict_accel(self, x, y, new_x, axis, dim=6):
        self.models[axis].set_train_data(inputs=torch.tensor(x), targets=torch.tensor(y).flatten(), strict=False)
        self.models[axis].covar_module.base_kernel.lengthscale = torch.tensor(self.length_hypers[axis])
        self.models[axis].covar_module.outputscale = torch.tensor(self.scale_hypers[axis])
        self.models[axis].likelihood.noise = torch.tensor(self.noise_variance_lin)

        if self.online_regression:
            self.models[axis].train()
            self.likelihoods[axis].train()
            optimizer = torch.optim.Adam(self.models[axis].parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[axis], self.models[axis])
            for _ in range(1):  # max_iters=1
                optimizer.zero_grad()
                output = self.models[axis](torch.tensor(x))
                loss = -mll(output, torch.tensor(y).flatten())
                loss.backward()
                optimizer.step()

            self.length_hypers[axis] = self.models[axis].covar_module.base_kernel.lengthscale.tolist()
            self.scale_hypers[axis] = self.models[axis].covar_module.outputscale.item()

        self.models[axis].eval()
        self.likelihoods[axis].eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihoods[axis](self.models[axis](torch.tensor(new_x)))
            mean = observed_pred.mean.numpy()
            var = observed_pred.variance.numpy()

        return mean, var