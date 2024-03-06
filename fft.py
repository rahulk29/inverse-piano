import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq

NOTE_THRESH = 0.001
TEMPO = 120  # bpm

notes = [
    {"name": "A2", "file": "a2_steinway.wav"},
    {"name": "A3", "file": "a3_steinway.wav"},
    {"name": "A4", "file": "a4_steinway.wav"},
]
file_path = "twinkle_twinkle_as.wav"


def window_size(sample_rate, tempo):
    return int(round(60 / tempo / 2 * sample_rate))


def aud_open(path):
    samples, sampling_rate = librosa.load(
        path, sr=None, mono=True, offset=0.0, duration=None
    )
    return samples, sampling_rate


def aud_fft(y, sampling_rate):
    n = len(y)
    T = 1 / sampling_rate

    yf = fft(y, norm="forward")[: n // 2]
    xf = fftfreq(n, T)[: n // 2]

    return xf, yf


def fft_plot(xf, yf):
    n = len(yf)
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / n * np.abs(yf))

    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()


samples, sampling_rate = aud_open(file_path)
duration = len(samples) / sampling_rate
print("Performing sophisticated analysis of input audio...")

window_size = window_size(sampling_rate, TEMPO)
n_windows = len(samples) // window_size

quarter_notes = []

for i in range(n_windows):
    window = samples[i * window_size : (i + 1) * window_size]
    xf, yf = aud_fft(window, sampling_rate)

    best_note, best_val = None, 0.0
    for note in notes:
        note_samples, note_sr = aud_open(note["file"])
        assert (
            note_sr == sampling_rate
        ), "note clips must have the same sampling rate as input audio"
        _, note_fft = aud_fft(note_samples[:window_size], note_sr)
        corr = np.correlate(yf, note_fft, mode="full") / np.sum(np.abs(note_fft))
        max_corr = np.max(np.abs(corr))
        # print(f'{note["name"]}: {max_corr}')
        if max_corr > NOTE_THRESH and max_corr > best_val:
            best_note = note["name"]
            best_val = max_corr
    if best_note is None:
        quarter_notes.append("X")
    else:
        quarter_notes.append(best_note)

print(f"NOTES: {quarter_notes}")
