import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Constants
c = 3e8  # Speed of light in m/s
wavelength = 1550e-9  # Optical wavelength in meters (1550 nm)
frequency = c / wavelength  # Frequency corresponding to 1550 nm
tx_power_range = np.arange(0.1, 1.01, 0.01)  # Transmit power from 100 mW to 1 W in steps of 10 mW
distances = [200e3, 400e3, 600e3, 800e3, 1000e3]  # Fixed distances in meters

# Beam characterization
def beam_divergence(w0, wavelength):
    return wavelength / (np.pi * w0)

def beam_spread(w0, z, wavelength):
    return w0 * np.sqrt(1 + (wavelength * z / (np.pi * w0**2))**2)

def intensity_profile(w0, z, I0, wavelength):
    wz = beam_spread(w0, z, wavelength)
    return I0 * (w0 / wz)**2

# Channel characterization (Free-space path loss)
def fspl(d, f):
    return (4 * np.pi * d * f / c)**2

def received_power(pt, d, f):
    return pt / fspl(d, f)

# Plot: Transmit Power vs Received Power
def plot_received_power():
    plt.figure(figsize=(10, 6))
    for d in distances:
        pr = [received_power(pt, d, frequency) for pt in tx_power_range]
        plt.plot(tx_power_range * 1e3, np.array(pr) * 1e3, label=f'{int(d/1e3)} km')
    plt.xlabel('Transmit Power (mW)')
    plt.ylabel('Received Power (mW)')
    plt.title('Transmit Power vs Received Power at Different Distances')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Q-function and BER for OOK
def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_ook(snr_linear):
    return Q(np.sqrt(2 * snr_linear))

# Plot: BER vs SNR for OOK modulation
def plot_ber_vs_snr():
    plt.figure(figsize=(10, 6))
    snr_db_range = np.arange(-2, 8, 1)
    pt = 0.2  # 200 mW
    for d in distances:
        pr = received_power(pt, d, frequency)
        bers = []
        for snr_db in snr_db_range:
            snr_linear = 10**(snr_db / 10)
            noise_power = pr / snr_linear
            actual_snr = pr / noise_power
            ber = ber_ook(actual_snr)
            bers.append(ber)
        plt.semilogy(snr_db_range, bers, marker='o', label=f'{int(d/1e3)} km')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR for OOK Modulation')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run both plots
plot_received_power()
plot_ber_vs_snr()
