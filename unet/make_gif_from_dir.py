import os
import glob
import tqdm
import imageio

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='Path to the image directory')
parser.add_argument('--fps', type=int, required=True, help='FPS of the gif')
args = vars(parser.parse_args())

extensions = ['jpg', 'jpeg', 'png', 'ppm']
images = []
image_paths = []

print('[INFO] Creating gif from image directory ...')
for ext in extensions:
    image_paths += glob.glob(os.path.join(args['dir'], f'*.{ext}'))

image_paths = {int(x.split('/')[-1].split('.')[0]):x for x in image_paths}

with tqdm.tqdm(total=len(image_paths)) as pbar:
    for filename in sorted(image_paths):
        images.append(imageio.imread(image_paths[filename]))
        pbar.update(1)

imageio.mimsave('gifs/output.gif', images, fps=args['fps'])
print('[INFO] GIF from image directory created')
