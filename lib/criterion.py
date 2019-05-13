import torch
from torch import nn
import torch.nn.functional as F


class ReconstructionCriterion(nn.Module):
    def __init__(self, sigma=1):
        super(ReconstructionCriterion, self).__init__()
        self.sigma = sigma

    def forward(self, original, reconstruct):
        batch_size = original.size(0)
        reconstruct_loss = F.mse_loss(reconstruct, original, reduction="sum")
        reconstruct_loss = reconstruct_loss / (self.sigma ** 2 * batch_size)
        return reconstruct_loss


class KLCriterion(nn.Module):
    def __init__(self):
        super(KLCriterion, self).__init__()

    def forward(self, z_mean, z_log_sigma, z_sigma):
        batch_size = z_mean.size(0)
        z_mean_sq = z_mean * z_mean
        z_sigma_sq = z_sigma * z_sigma
        z_log_sigma_sq = 2 * z_log_sigma
        kl_loss = torch.sum(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1) / batch_size
        return kl_loss


class ClassificationCriterion(nn.Module):
    def __init__(self):
        super(ClassificationCriterion, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, predict, gt):
        return self.criterion(predict, gt)


class VAECriterion(nn.Module):
    """
    Here we calculate the VAE loss
    VAE loss's math formulation is :
    E_{z~Q}[log(P(X|z))]-D[Q(z|X)||P(z)]
    which can be transformed into:
    ||X-X_{reconstructed}||^2/(\sigma)^2 - [<L2norm(u)>^2+<L2norm(diag(\Sigma))>^2
    -<L2norm(diag(ln(\Sigma)))>^2-1]
    Our input is :
    x_sigma,x_reconstructed,x,z_mean,z_Sigma
    """

    def __init__(self, x_sigma=1):
        super(VAECriterion, self).__init__()
        self.x_sigma = x_sigma

    def forward(self, x, x_reconstructed, z_mean, z_log_sigma, z_sigma):
        """
        :param x: input & ground truth
        :param x_reconstructed: the reconstructed output by VAE
        :param z_mean: the mean of latent space Q(z|X)
        :param z_sigma: the variance of latent space
        :param z_log_sigma : log(z_sigma)
        :return: reconstruct_loss, kl_loss
        """
        batch_size = x.size(0)
        # calculate reconstruct loss, sum in instance, mean in batch
        reconstruct_loss = F.mse_loss(x_reconstructed, x, reduction="sum")
        reconstruct_loss = reconstruct_loss / (self.x_sigma ** 2 * batch_size)
        # reconstruct_loss = F.mse_loss(x_reconstructed, x)
        # reconstruct_loss = reconstruct_loss / self.x_sigma ** 2
        # calculate latent space KL divergence
        z_mean_sq = z_mean * z_mean
        z_sigma_sq = z_sigma * z_sigma
        z_log_sigma_sq = 2 * z_log_sigma
        kl_loss = torch.sum(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1) / batch_size
        # kl_loss = 0.5 * torch.mean(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1)
        # notice here we duplicate the 0.5 by each part
        return reconstruct_loss, kl_loss
