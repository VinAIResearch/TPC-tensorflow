import numpy as np
import os
from PIL import Image
import wrappers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

size = (256, 256)
task = 'cheetah_run'
bg_path = './videos/train'
img_source_type = 'video'
max_videos = 100
random_bg = False
action_repeat = 5
time_limit = 1000
steps = 50
img_len = 5

resource_files = [
    os.path.join(bg_path, f) for f in os.listdir(bg_path)
    if os.path.isfile(os.path.join(bg_path, f))
]
env = wrappers.DeepMindControl(task, size, resource_files=resource_files, img_source=img_source_type, random_bg=random_bg)
env = wrappers.ActionRepeat(env, action_repeat)

env.reset()
new_im = Image.new('RGB', (size[0]*img_len, size[0]))
x_offset = 0
for step in range(steps):
    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
    if step >= steps - img_len:
        new_im.paste(Image.fromarray(obs['image'], 'RGB'), (x_offset,0))
        x_offset += size[0]
    
new_im.save('natural_bg.jpg')