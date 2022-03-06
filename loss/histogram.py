from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HistLayer(nn.Module):
    """Deep Neural Network Layer for Computing Differentiable Histogram.

    Computes a differentiable histogram using a hard-binning operation implemented using
    CNN layers as desribed in `"Differentiable Histogram with Hard-Binning"
    <https://arxiv.org/pdf/2012.06311.pdf>`_.

    code:https://github.com/Alikerin/AMMI-Project

    Attributes:
        in_channel (int): Number of image input channels.
        numBins (int): Number of histogram bins.
        learnable (bool): Flag to determine whether histogram bin widths and centers are
            learnable.
        centers (List[float]): Histogram centers.
        widths (List[float]): Histogram widths.
        two_d (bool): Flag to return flattened or 2D histogram.
        bin_centers_conv (nn.Module): 2D CNN layer with weight=1 and bias=`centers`.
        bin_widths_conv (nn.Module): 2D CNN layer with weight=-1 and bias=`width`.
        threshold (nn.Module): DNN layer for performing hard-binning.
        hist_pool (nn.Module): Pooling layer.
    """

    def __init__(self, in_channels, num_bins=4, two_d=False):
        super(HistLayer, self).__init__()

        # histogram data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.learnable = False
        bin_edges = np.linspace(-0.05, 1.05, num_bins + 1)
        centers = bin_edges + (bin_edges[2] - bin_edges[1]) / 2
        self.centers = centers[:-1]
        self.width = (bin_edges[2] - bin_edges[1]) / 2
        self.two_d = two_d

        # prepare NN layers for histogram computation
        self.bin_centers_conv = nn.Conv2d(
            self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.in_channels,
            bias=True,
        )
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        self.bin_centers_conv.bias.data = torch.nn.Parameter(
            -torch.tensor(self.centers, dtype=torch.float32)
        )
        self.bin_centers_conv.bias.requires_grad = self.learnable

        self.bin_widths_conv = nn.Conv2d(
            self.numBins * self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.numBins * self.in_channels,
            bias=True,
        )
        self.bin_widths_conv.weight.data.fill_(-1)
        self.bin_widths_conv.weight.requires_grad = False
        self.bin_widths_conv.bias.data.fill_(self.width)
        self.bin_widths_conv.bias.requires_grad = self.learnable

        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight
        self.threshold = nn.Threshold(1, 0)
        self.hist_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, normalize=True):
        """Computes differentiable histogram.
        Args:
            input_image: input image.
        Returns:
            flattened and un-flattened histogram.
        """
        # |x_i - u_k|
        xx = self.bin_centers_conv(input_image)
        xx = torch.abs(xx)

        # w_k - |x_i - u_k|
        xx = self.bin_widths_conv(xx)

        # 1.01^(w_k - |x_i - u_k|)
        xx = torch.pow(torch.empty_like(xx).fill_(1.01), xx)

        # Φ(1.01^(w_k - |x_i - u_k|), 1, 0)
        xx = self.threshold(xx)

        # clean-up
        two_d = torch.flatten(xx, 2)
        if normalize:
            xx = self.hist_pool(xx)
        else:
            xx = xx.sum([2, 3])
        one_d = torch.flatten(xx, 1)
        return one_d, two_d


def emd_loss(hgram1, hgram2):
    """Computes Earth Mover's Distance (EMD) between histograms

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        EMD loss.
    """
    return (
        ((torch.cumsum(hgram1, dim=1) - torch.cumsum(hgram2, dim=1)) ** 2).sum(1).mean()
    )


def mae_loss(histogram_1: Tensor, histogram_2: Tensor) -> Tensor:
    """Computes Mean Absolute Error (MAE) between histograms

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        MAE loss.
    """
    return (torch.abs(histogram_1 - histogram_2)).sum(1).mean(-1).mean()


def mse_loss(histogram_1: Tensor, histogram_2: Tensor) -> Tensor:
    """Computes Mean Squared Error (MSE) between histograms.

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        MSE loss.
    """
    return torch.pow(histogram_1 - histogram_2, 2).sum(1).mean(-1).mean()


class HistogramLoss(nn.Module):
    def __init__(self, loss_fn, num_bins, rgb=False, yuv=True, yuvgrad=False):
        super().__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.histlayer = HistLayer(in_channels=1, num_bins=num_bins)
        loss_dict = {"emd": emd_loss, "mae": mae_loss, "mse": mse_loss}
        self.loss_fn = loss_dict[loss_fn]

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def to_YUV(self, image):
        """Converts image from RGB to YUV color space.

        Arguments:
            image: batch of images with shape (batch_size x num_channels x width x height).

        Returns:
            batch of images in YUV color space with shape
            (batch_size x num_channels x width x height).
        """
        y = (
            (0.299 * image[:, 0, :, :])
            + (0.587 * image[:, 1, :, :])
            + (0.114 * image[:, 2, :, :])
        )
        u = (
            (-0.14713 * image[:, 0, :, :])
            + (-0.28886 * image[:, 1, :, :])
            + (0.436 * image[:, 2, :, :])
        )
        v = (
            (0.615 * image[:, 0, :, :])
            + (-0.51499 * image[:, 1, :, :])
            + (-0.10001 * image[:, 2, :, :])
        )
        image = torch.stack([y, u, v], 1)
        return image

    def extract_hist(self, image, one_d=False, normalize=False):
        """Extracts both vector and 2D histogram.

        Args:
            layer: histogram layer.
            image: input image tensor, shape: batch_size x num_channels x width x height.

        Returns:
            list of tuples containing 1d (and 2d histograms) for each channel.
            1d histogram shape: batch_size x num_bins
            2d histogram shape: batch_size x num_bins x width*height
        """
        # comment these lines when you inputs and outputs are in [0,1] range already
        image = (image + 1) / 2
        _, num_ch, _, _ = image.shape
        hists = []
        for ch in range(num_ch):
            hists.append(
                self.histlayer(image[:, ch, :, :].unsqueeze(1), normalize=normalize)
            )
        if one_d:
            return [one_d_hist for (one_d_hist, _) in hists]
        return hists

    @staticmethod
    def entropy(histogram):
        """Compute Shannon Entropy"""
        samples = []
        for sample in histogram:
            # Remove zeros
            sample = sample[sample > 0]
            result = -torch.sum(sample * torch.log(sample)).unsqueeze(0)
            samples.append(result)
        return torch.cat(samples)

    def dmi(self, hgram1, hgram2):
        """Compute Mutual Information metric.

        Arguments:
            hgram1: 2D histogram for image_1, shape: batch_size x num_bins x height*width
            hgram2: 2D histogram for image_2, shape: batch_size x num_bins x height*width

        Return:
            Returns `1 - MI(I_1, I_2)/Entropy(I_1, I_2)`
        """
        # compute joint histogram and marginals
        pxy = torch.bmm(hgram1, hgram2.transpose(1, 2)) / hgram1.shape[-1]
        px = torch.sum(pxy, axis=1)  # marginal for x over y
        py = torch.sum(pxy, axis=2)  # marginal for y over x
        joint_entropy = self.entropy(pxy)
        mi = self.entropy(px) + self.entropy(py) - joint_entropy
        return torch.mean(1 - (mi / joint_entropy))

    def hist_loss(
        self, histogram_1: List[Tensor], histogram_2: List[Tensor]
    ) -> Tuple[float, float]:
        """Compute Histogram Losses.

        Computes EMD and MI losses for each channel, then returns the mean.

        Args:
            histogram_1: first histogram tensor, shape: batch_size x num_channels x num_bins.
            histogram_1: second histogram tensor, shape: batch_size x num_channels x num_bins
            loss_type: type of loss function.

        Returns:
            Tuple containing mean of EMD and MI losses respectively.
        """
        emd = 0
        mi = 0
        num_channels = 0
        for channel_hgram1, channel_hgram2 in zip(histogram_1, histogram_2):
            emd += self.loss_fn(channel_hgram1[0], channel_hgram2[0])
            mi += self.dmi(channel_hgram1[1], channel_hgram2[1])
            num_channels += 1
        return emd / num_channels, mi / num_channels

    def __call__(self, input, reference):

        emd_total_loss = 0
        mi_total_loss = 0
        if self.rgb:
            emd, mi = self.hist_loss(
                self.extract_hist(input), self.extract_hist(reference)
            )
            emd_total_loss += emd
            mi_total_loss += mi
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            emd, mi = self.hist_loss(
                self.extract_hist(input_yuv), self.extract_hist(reference_yuv)
            )
            emd_total_loss += emd
            mi_total_loss += mi
        if self.yuvgrad:
            input_v, input_h = self.get_image_gradients(input_yuv)
            ref_v, ref_h = self.get_image_gradients(reference_yuv)

            emd, mi = self.hist_loss(
                self.extract_hist(input_v), self.extract_hist(ref_v)
            )
            emd_total_loss += emd
            mi_total_loss += mi
            emd, mi = self.hist_loss(
                self.extract_hist(input_h), self.extract_hist(ref_h)
            )
            emd_total_loss += emd
            mi_total_loss += mi

        return emd_total_loss, mi_total_loss
