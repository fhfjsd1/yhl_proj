# %%

import torch
import torchaudio
import numpy as np

import matplotlib.pyplot as plt

def pre_emphasis(signal, coeff=0.97):
    """
    Apply pre-emphasis filter to the signal.
    y[n] = x[n] - coeff * x[n-1]
    signal: torch.Tensor, shape (num_channels, num_samples)
    Returns a tensor of the same shape.
    """
    # Ensure signal is float tensor
    if not torch.is_floating_point(signal):
        signal = signal.float()
    # Compute pre-emphasis along the time dimension (i.e., last dimension)
    return torch.cat((signal[:, :1], signal[:, 1:] - coeff * signal[:, :-1]), dim=1)

def plot_filter_frequency_response(coeff=0.97, num_points=512):
    """
    Plot the magnitude frequency response of the pre-emphasis filter H(e^(jw)) = 1 - coeff * e^(-jw)
    """
    # Frequency axis from 0 to pi (normalized frequency)
    omega = np.linspace(0, np.pi, num_points)
    # Frequency response: H(e^(jw)) = 1 - coeff * exp(-j omega)
    H = 1 - coeff * np.exp(-1j * omega)
    magnitude = np.abs(H)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
    
    plt.figure(figsize=(8, 4))
    plt.plot(omega / np.pi, magnitude)
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Magnitude')
    plt.title('Pre-Emphasis Filter Frequency Response')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrograms(original, processed, sample_rate, n_fft=400, hop_length=200):
    """
    Plot spectrograms for original and processed signals.
    Both original and processed are torch.Tensors with shape (num_channels, num_samples).
    """
    # Take the first channel for display
    original = original[0]
    processed = processed[0]
    
    # Create a Spectrogram transform from torchaudio
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=hop_length)
    # Compute spectrograms
    original_spec = spectrogram_transform(original)
    processed_spec = spectrogram_transform(processed)
    
    # Quantitatively evaluate pre-emphasis effectiveness by comparing the high-frequency energy ratio.
    # Compute frequency bins corresponding to the spectrogram rows.
    freqs = torch.linspace(0, sample_rate / 2, steps=original_spec.shape[0])
    # Define a threshold frequency (e.g., one-fourth of the sampling rate)
    threshold = sample_rate / 4
    high_freq_mask = freqs >= threshold

    # Compute energy per spectrogram: square magnitude
    orig_energy = original_spec.pow(2)
    proc_energy = processed_spec.pow(2)
    
    # Sum total energy over all frequencies and time frames
    orig_total_energy = orig_energy.sum()
    proc_total_energy = proc_energy.sum()
    
    # Sum energy in the high-frequency band only (along frequency axis)
    orig_high_energy = orig_energy[high_freq_mask, :].sum()
    proc_high_energy = proc_energy[high_freq_mask, :].sum()
    
    # Calculate high-frequency energy ratio
    orig_ratio = orig_high_energy / orig_total_energy
    proc_ratio = proc_high_energy / proc_total_energy
    
    print(f"High-frequency energy ratio (original): {orig_ratio:.4f}")
    print(f"High-frequency energy ratio (pre-emphasized): {proc_ratio:.4f}")
    
    # Convert to dB scale
    original_db = 20 * torch.log10(torch.clamp(original_spec, min=1e-10))
    processed_db = 20 * torch.log10(torch.clamp(processed_spec, min=1e-10))
    
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    
    im0 = axs[0].imshow(original_db.numpy(), origin="lower", aspect="auto",
                         extent=[0, original.size(0)/sample_rate, 0, sample_rate/2],
                         cmap="viridis")
    axs[0].set_title("Original Audio Spectrogram")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(processed_db.numpy(), origin="lower", aspect="auto",
                         extent=[0, processed.size(0)/sample_rate, 0, sample_rate/2],
                         cmap="viridis")
    axs[1].set_title("Pre-Emphasized Audio Spectrogram")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.show()

def main():
    # Set the path to your audio file (update the path as needed)
    # audio_path = r"American_Female.wav"
    audio_path = r"00172010092500051674x.wav"

    
    # Load audio using torchaudio
    signal, sample_rate = torchaudio.load(audio_path)  # signal shape: (channels, samples)
    
    # Plot the frequency response of the pre-emphasis filter
    plot_filter_frequency_response(coeff=0.97)
    
    # Apply pre-emphasis
    emphasized_signal = pre_emphasis(signal, coeff=0.97)
    
    # Plot spectrograms before and after pre-emphasis
    plot_spectrograms(signal, emphasized_signal, sample_rate)

if __name__ == '__main__':
    main()
# %%