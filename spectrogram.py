import os

def make_single_spectrogram(input_filepath, output_filepath):
    os.system(f'sox "{input_filepath}" -n spectrogram -Y 200 -X 50 -m -r -o {output_filepath}')

def make_spectrograms():
    cwd = os.getcwd()
    genres_folder = os.path.join(cwd, 'genres')
    spectrogram_folder = os.path.join(cwd, 'spectrograms')
    try:
        os.mkdir(spectrogram_folder)
    except FileExistsError:
        pass 
    
    for subdir, dirs, files in os.walk(genres_folder):
        for file in files:
            if file.endswith('.au'):
                genre_folder = os.path.join(spectrogram_folder, os.path.basename(subdir))
                try:
                    os.mkdir(genre_folder)
                except FileExistsError:
                    pass 

                current_filepath = os.path.join(subdir, file)
                spectrogram_output = os.path.join(genre_folder, f'{file}.png')
                make_single_spectrogram(current_filepath, spectrogram_output)


if __name__ == '__main__':
    make_spectrograms()