# 129x150
from PIL import Image
import os

def slice_image(filepath, filename, target_folder):
    height = 128
    width = 128

    img = Image.open(filepath)
    imgwidth, imgheight = img.size
    slices = []
    for j in range(0, imgwidth-width, width):
        box = (j, 0, j + width, 0 + height)
        a = img.crop(box)
        try:
            o = a
            strfile = filename.rstrip(".png")
            savepath = os.path.join(target_folder, strfile + "_" + str(j) + ".png")
            o.save(savepath)
            slices.append(savepath)
        except Exception as e:
            print(e)
    return slices 

def make_slices():
    cwd = os.getcwd()
    spectrogram_folder = os.path.join(cwd, 'spectrograms')
    slices_folder = os.path.join(cwd, 'dataset_photos', 'slices')
    try:
        os.mkdir(slices_folder)
    except FileExistsError:
        pass

    for subdir, dirs, files in os.walk(spectrogram_folder):
        for file in files:
            if file.endswith('.png'):
                spectrogram_folder = os.path.join(slices_folder, os.path.basename(subdir))
                try:
                    os.mkdir(spectrogram_folder)
                except FileExistsError:
                    pass

                current_filepath = os.path.join(subdir, file)
                slice_image(current_filepath, file, spectrogram_folder)
                


if __name__ == '__main__':
    make_slices()
