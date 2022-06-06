import os
import re
import glob
from pathlib import Path
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from PIL import Image
from functools import partial
from tqdm.auto import tqdm
import typer
from pqdm.processes import pqdm

app = typer.Typer()


def crop(image, target_size = (1024, 1024)):
    crop_x = 168
    crop_y = 38

    if crop_y:
        img = image[crop_y:-crop_y, 553 + crop_x:-27 - crop_x, :]
    else:
        img = image[:, 550 + crop_x:-25 - crop_x, :]

    img = cv2.resize(img, dsize=target_size) # , interpolation=cv2.INTER_CUBIC)
    return img


def filter():
        img_dir = '/shared/results/Skopia/videos_8frames'
        duration = 8
        
        images = os.listdir(img_dir)
        img_name = re.compile('([0-9]+_[0-9]+)_[0-7]+.jpg')
        video_names = {img_name.search(img).groups()[0] for img in images}
        video_names = list(video_names)
        video_names = ['19549_27023', '19104_16476', '19188_6691'] # ['19106_13238', '19350_21358', '19481_8649', '19549_27023']
        for i, video_name in enumerate(video_names):
            print(f'video {i} of {len(video_names)} | {video_name}')

            img_path = img_dir + f'/{video_name}_{0}.jpg'
            img = np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8) # .transpose(2, 0, 1)
            prev_img = img.copy()
            counter = 0
            for t in range(1, duration):
                img_path = img_dir + f'/{video_name}_{t}.jpg'
                img = np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8) # .transpose(2, 0, 1)
                # im = Image.fromarray(img-prev_img)
                # im.save("test_img.jpeg")
                if np.allclose(img, prev_img, rtol=5, atol=1):
                    counter += 1
                prev_img = img.copy()
            if counter > 0:
                print('\t', counter)



def process_video(
    video_path: Path, 
    img_dir: Path, 
    output_dir: Path, 
    length: int = 8, 
    fts: int = 5, 
    distance: int = 4, 
    target_size: tuple = (1024, 1024)
):
    img_name = re.compile('([0-9]+)_([0-9]+).jpg')   # image name template
    pos = length // 2 - 1

    video_id = video_path.split('/')[-2]
    video = cv2.VideoCapture(video_path)
    fc = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_frame_num = 0
    sharp_images = glob.glob(f'{img_dir}/{video_id}*.jpg')
    sharp_images = sorted(sharp_images, key=lambda x: int(img_name.search(x).groups()[-1]))

    for img in tqdm(sharp_images):
        video_id, frame_num = img_name.search(img).groups()
        frame_num = int(frame_num)
        
        if frame_num in range(pos*fts, fc-distance*fts) and \
            frame_num - prev_frame_num > distance*fts:
            prev_frame = 0
            for i in range(-length // 2 + 1, length // 2 + 1):
                frame_id = frame_num + i*fts
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = video.read()
                # import numpy as np
                # if np.allclose(frame, prev_frame, rtol=50, atol=50):
                #     print('gotcha')
                if not ret:
                    print(img, frame_id)
                frame_path = output_dir + f'/{video_id}_{frame_num}_{i+length // 2}.jpg'
                frame = crop(frame, target_size)
                cv2.imwrite(frame_path, frame)
                prev_frame=frame.copy()

            prev_frame_num = frame_num       
    video.release()


@app.command()
def create_dataset(
    video_dir: str = typer.Option(..., "-v", help="Directory with video images"),
    img_dir: str = typer.Option(..., "-i", help="Directory with sharp images"),
    output_dir: str = typer.Option(..., "-o", help="Output directory for saved frames"),
    length: int = typer.Option(8, "-l", help="Length of extracted videos"),
    target_size: int = typer.Option(1024, "-s", help="Size of each frame"),
    distance: int = typer.Option(4, "-d", help="Acceptable distance between two following sharp images"),
    fts: int = typer.Option(5, "-f", help="frames to skip"),
    n_jobs: int = typer.Option(8, "-j", help="Number of job processes spawned.")
):
    target_size = (target_size, target_size)

    videos = glob.glob(video_dir + '*/*.mp4')
    run_batch = partial(
        process_video, 
        img_dir=img_dir, 
        output_dir=output_dir, 
        length=length, 
        fts=fts, 
        distance=distance, 
        target_size=target_size
    )
    pqdm(videos, run_batch, n_jobs=n_jobs)


if __name__ == "__main__":
    app()



# if __name__ == '__main__':
#     # filter()
#     # import sys
#     # sys.exit()
#     video_dir = '/shared/results/Skopia/data/'
#     img_dir = '/shared/results/Skopia/images2'
#     output_dir = '/shared/results/Skopia/videos_24frames'
#     img_name = re.compile('([0-9]+)_([0-9]+).jpg')   # image name template

#     length = 24  # length of extracted videos
#     pos = length // 2 - 1      # sharp image position (center)
#     d = 4        # acceptable distance between two following sharp images
#     fts = 5      # frames to skip
#     target_size = (256, 256)

#     videos = glob.glob(video_dir + '*/*.mp4')
#     for video_path in tqdm(videos):
#         video_id = video_path.split('/')[-2]
#         video = cv2.VideoCapture(video_path)
#         fc = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#         prev_frame_num = 0
#         sharp_images = glob.glob(f'{img_dir}/{video_id}*.jpg')
#         sharp_images = sorted(sharp_images, key=lambda x: int(img_name.search(x).groups()[-1]))

#         for img in tqdm(sharp_images):
#             video_id, frame_num = img_name.search(img).groups()
#             frame_num = int(frame_num)
            
#             if frame_num in range(pos*fts, fc-d*fts) and \
#                 frame_num - prev_frame_num > d*fts:
#                 prev_frame = 0
#                 for i in range(-length // 2 + 1, length // 2 + 1):
#                     frame_id = frame_num + i*fts
#                     video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#                     ret, frame = video.read()
#                     # import numpy as np
#                     # if np.allclose(frame, prev_frame, rtol=50, atol=50):
#                     #     print('gotcha')
#                     if not ret:
#                         print(img, frame_id)
#                     frame_path = output_dir + f'/{video_id}_{frame_num}_{i+length // 2}.jpg'
#                     frame = crop(frame, target_size)
#                     cv2.imwrite(frame_path, frame)
#                     prev_frame=frame.copy()

#                 prev_frame_num = frame_num       
#         video.release()
