import time
import pandas as pd
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

def memory_usage(name, df):
    memory = df.memory_usage(deep=True).sum()
    if memory < 1024:
        print(f'{name} Memory Usage: {memory:.2f} bytes')
        return
    if memory < 1024**2:
        print(f'{name} Memory Usage: {memory/1024:.2f} KB')
        return
    if memory < 1024**3:
        print(f'{name} Memory Usage: {memory/1024**2:.2f} MB')
        return
    print(f'{name} Memory Usage: {memory/1024**3:.2f} GB')

def fourier_transform(sound):
    if sound.channels == 2:
        sound = sound.split_to_mono()[0]
    fourier = np.fft.fft(sound.get_array_of_samples())
    n = len(fourier)
    frequencies = np.fft.fftfreq(n, 1/sound.frame_rate)
    return frequencies, fourier

def group_frequencies_rms(frequencies, fourier, min_freq=20, max_freq=20000, num_bins=120):
    log_bins = np.logspace(np.log10(min_freq), np.log10(max_freq), num=num_bins)
    bin_indices = np.digitize(frequencies, log_bins) - 1  # Get bin indices for each frequency
    log_rms_spectrum = np.zeros(num_bins - 1)

    for i in range(num_bins - 1):
        bin_values = fourier[bin_indices == i]
        if bin_values.size > 0:
            log_rms_spectrum[i] = np.sqrt(np.mean(np.abs(bin_values)**2))

    return log_bins[:-1], log_rms_spectrum

def process_row(row):
    row_start_time = time.time()
    print('loading:', row['filename'])
    print(row)
    sound = AudioSegment.from_file(f"Data/birdsong-recognition/train_audio/{row['ebird_code']}/{row['filename']}", format="mp3")
    frequencies, fourier = fourier_transform(sound)
    log_bins, log_rms_spectrum = group_frequencies_rms(frequencies, fourier)
    result = {}
    for j, bin_value in enumerate(log_bins[:-1]):
        column_name = f"{bin_value:.0f}Hz"
        result[column_name] = log_rms_spectrum[j]
    print('Processed:', row['filename'])
    print(f'Time taken: {time.time() - row_start_time:.2f} seconds')
    return row.name, result


def main():
    df = pd.read_csv('Data/birdsong-recognition/train.csv')
    taxonomy = pd.read_csv('Data/ebird_taxonomy_v2023.csv')
    df = df.merge(taxonomy, left_on='ebird_code', right_on='SPECIES_CODE', how='left', indicator=True)
    df = df.drop(columns=['_merge', 'playback_used', 'pitch', 'duration', 'speed',
                        'species', 'number_of_notes', 'title', 'secondary_labels',
                        'bird_seen', 'sci_name', 'location', 'description',
                        'bitrate_of_mp3', 'volume', 'background', 'xc_id',
                        'url', 'country', 'author', 'primary_label',
                        'length', 'recordist', 'license', 'TAXON_ORDER',
                        'CATEGORY', 'SPECIES_CODE', 'TAXON_CONCEPT_ID', 'REPORT_AS'])
    df.rename(columns={'PRIMARY_COM_NAME': 'common_name', 
                    'SCI_NAME': 'sci_name', 
                    'ORDER': 'order', 
                    'FAMILY': 'family', 
                    'SPECIES_GROUP': 'species_group'}, inplace=True)
    log_bins = np.logspace(np.log10(20), np.log10(20000), num=120)

    # Add columns to the dataframe for each element in log_bins
    #memory_usage('Data before Frequency Bins', df)
    for bin_value in log_bins:
        column_name = f"{bin_value:.0f}Hz"
        df[column_name] = np.nan  # Initialize with NaN or any default value
        df = df.copy()  # Force pandas to allocate memory for the new column
    print(df.head())
    
    print('-------------------BEGINING PROCESSING-------------------')

    # Use ThreadPoolExecutor to process rows in parallel
    start_time = time.time()
    completed_rows = 0  # Counter for completed rows

    for i, row in df.iterrows():
        process_row(row)
        completed_rows += 1
        percent_complete = completed_rows / len(df)
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / percent_complete
        estimated_time_remaining = estimated_total_time - elapsed_time
        hours, rem = divmod(estimated_time_remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        time_remaining_str = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'
        print(f'Estimated time remaining: {time_remaining_str}')
        print(f'{percent_complete:.2%} complete')
        print('---')

    #memory_usage('Data after filling Frequency Bins', df)
    df.to_csv('processed_birdsong_data.csv', index=False)


if __name__ == '__main__':
    main()