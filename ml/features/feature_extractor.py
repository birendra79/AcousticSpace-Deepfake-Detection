import librosa

from ml.dataset.audio_reader import load_audio
from ml.preprocessing.audio_preprocessor import preprocess_audio


def extract_mfcc(audio, sample_rate):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    return mfcc


def plot_mfcc(mfcc, sample_rate):
    # Imported here (not at module top) so the live server doesn't need
    # matplotlib/librosa.display just to import this file.
    import librosa.display
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mfcc,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    
    plt.show()