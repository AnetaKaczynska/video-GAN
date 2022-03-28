import glob

from moviepy.editor import *


images = glob.glob(r'/home/z1143165/video-GAN/output_networks/jelito3d_batchsize8/jelito3d_batchsize8_s8_i200000_interpolations/?_?/*')
images = sorted(images)
print('generating')
for img in images:
	print(img)
clip = ImageSequenceClip(images, fps=30)
print(clip.duration)
print('saving')
clip.write_videofile("0-1-2-3.mp4", fps=30, audio=False)
