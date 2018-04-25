# 129x150
from PIL import Image
import os


def make_slices():
    height = 129
    width = 150

    cwd = os.getcwd()
    spectrogram_folder = os.path.join(cwd, 'spectrograms')
    slices_folder = os.path.join(cwd, 'slices')
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

                # slice
                img = Image.open(current_filepath)
                imgwidth, imgheight = img.size
                for i in range(0, imgheight, height):
                    for j in range(0, imgwidth-width, width):
                        box = (j, i, j + width, i + height)
                        a = img.crop(box)
                        try:
                            o = a
                            strfile = file.rstrip(".au.png")
                            o.save(os.path.join(spectrogram_folder, strfile + "_" + str(j) + ".png"))
                        except Exception as e:
                            print(e)


if __name__ == '__main__':
    make_slices()
