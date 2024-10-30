from scipy import signal
import numpy as np
import torch

class DataManipulator:
    r"""
        Utility class. Refactored out of `ICMSDataset` to keep that focused to dataloading.
    """

    @staticmethod
    def kernel_smooth(
        spikes: torch.Tensor,
        window
    ) -> torch.Tensor:
        window = torch.tensor(window).float()
        window /=  window.sum()
        # Record B T C
        b, t, c = spikes.size()
        spikes = spikes.permute(0, 2, 1).reshape(b*c, 1, t).float()
        # Convolve window (B 1 T) with record as convolution will sum across channels.
        window = window.unsqueeze(0).unsqueeze(0)
        smooth_spikes = torch.nn.functional.conv1d(spikes, window, padding="same")
        return smooth_spikes.reshape(b, c, t).permute(0, 2, 1)

    @staticmethod
    def gauss_smooth(
        spikes: torch.Tensor,
        bin_size: float,
        kernel_sd=0.05,
        window_deviations=7, # ! Changed bandwidth from 6 to 7 so there is a peak
        past_only=False
    ) -> torch.Tensor:
        r"""
            Compute Gauss window and std with respect to bins

            kernel_sd: in seconds
            bin_size: in seconds
            past_only: Causal smoothing, only wrt past bins - we don't expect smooth firing in stim datasets as responses are highly driven by stim.
        """
        # input b t c
        gauss_bin_std = kernel_sd / bin_size
        # the window extends 3 x std in either direction
        win_len = int(window_deviations * gauss_bin_std)
        # Create Gaussian kernel
        window = signal.windows.gaussian(win_len, gauss_bin_std, sym=True)
        if past_only:
            window[len(window) // 2 + 1:] = 0 # Always include current timestep
            # if len(window) % 2:
            # else:
                # window[len(window) // 2 + 1:] = 0
        return DataManipulator.kernel_smooth(spikes, window)


r"""
    H1 filtering
"""
NEURAL_TAU_MS = 240. # exponential filter from H1 Lab
def apply_exponential_filter(
        x, tau=NEURAL_TAU_MS, bin_size=20, extent: int=1
    ):
    """
    Apply a **causal** exponential filter to the neural signal.

    :param x: NumPy array of shape (time, channels)
    :param tau: Decay rate (time constant) of the exponential filter
    :param bin_size: Bin size in ms (default is 10ms)
    :return: Filtered signal
    :param extent: Number of time constants to extend the filter kernel

    Implementation notes:
    # extent should be 3 for reporting parity, but reference hardcodes a kernel that's equivalent to extent=1
    """
    t = np.arange(0, extent * tau, bin_size)
    # Exponential filter kernel
    kernel = np.exp(-t / tau)
    kernel /= np.sum(kernel)
    # Apply the filter
    filtered_signal = np.array([signal.convolve(x[:, ch], kernel, mode='full')[:len(x)] for ch in range(x.shape[1])]).T
    return filtered_signal


def zscore_data(data): 
    """
    Z-scores the input data.

    Parameters:
    data (np.ndarray): The input data.

    Returns:
    np.ndarray: The z-scored data.
    """
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    s[s==0] = 1
    return (data - m) / s

def generate_lagged_matrix(input_matrix: np.ndarray, lag: int):
    """
    Generate a lagged version of an input matrix; i.e. include history in the input matrix.

    Parameters:
    input_matrix (np.ndarray): The input matrix. T x H
    lag (int): The number of lags to consider.
    zero_pad (bool): Whether to zero pad the lagged matrix to match input

    Returns:
    np.ndarray: The lagged matrix, shape T x (H * (lag + 1))
    """
    if lag == 0:
        return input_matrix
    # Initialize the lagged matrix
    lagged_matrix = np.zeros((input_matrix.shape[0], input_matrix.shape[1], lag + 1))
    lagged_matrix[:, :, 0] = input_matrix
    # Fill the lagged matrix
    for i in range(lag + 1):
        lag_entry = np.roll(input_matrix, i, axis=0)
        lag_entry[:i] = 0
        lagged_matrix[:, :, i] = lag_entry
    lagged_matrix = lagged_matrix.reshape(input_matrix.shape[0], input_matrix.shape[1] * (lag + 1))
    return lagged_matrix