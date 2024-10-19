import time
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cupy as cp
import cupyx as cpx
from pydub import AudioSegment


# Function to print a progress bar that updates in place
def print_progress_bar(iteration, length, message=''):
    progress = iteration / length
    progress = int(progress * 100)
    # Determine color based on progress
    if progress < 25:
        color = '\033[91m'  # Red
    elif progress < 90:
        color = '\033[93m'  # Yellow
    else:
        color = '\033[92m'  # Green
    bar = color + '=' * progress + '\033[0m' + ' ' * (100 - progress)
    print('\033[?25l', end='')  # Hide cursor - causes issues in Jupyter
    print(f'\r[{bar}] - {iteration}/{length}', end='\033[K')
    if message:
        print(f' - {message}', end='\033[K')
    if progress == 100:
        print()
        print('\033[?25h', end='')  # Show cursor - causes issues in Jupyter
    else:
        print('\033[?25h', end='')  # Show cursor after each update - causes issues in Jupyter

def main():
    df = pd.read_csv('data/birdsong-recognition/train.csv')
    test_df = df.head(100)
    with cpx.profiler.profile():
      count = 0
      print("--------------------Starting CuPy Test--------------------")
      start_time = time.time()
      for _, row in test_df.iterrows():
          count += 1
          sound = AudioSegment.from_file(f"data/birdsong-recognition/train_audio/{row.ebird_code}/{row.filename}", format="mp3")
          samples = cp.array(sound.get_array_of_samples())
          cpx.scipy.fft.fft(samples)
          cpx.scipy.fft.fftfreq(len(samples), 1/sound.frame_rate)
          print_progress_bar(count, len(test_df), message=f"Loading {row.ebird_code}/{row.filename}: {row.duration} seconds")
      cupy_time = time.time() - start_time
      print('--------------------CuPy Test Complete--------------------')
      print(f'Time taken: {cupy_time:.2f} seconds')


if __name__ == '__main__':
    main()