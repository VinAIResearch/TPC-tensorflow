### This code generates video backgrounds that are used in the simplistic background setting
### videos for training and testing will be saved in videos/rain_train and videos/rain_test respectively

import cv2
import numpy as np
from tqdm import tqdm, trange
import os

SLANT_MIN = -3
SLANT_MAX = 3 + 1
LENGTH_MIN = 1
LENGTH_MAX = 2 + 1
WIDTH_MIN = 1
WIDTH_MAX = 2 + 1
SPEED_MIN = 1
SPEED_MAX = 2 + 1
DENSITY_MIN = 50
DENSITY_MAX = 100 + 1

def generate(imshape=(64, 64, 3), slant=1, drop_length=2, drop_width=1, drop_color=(255, 255, 255), speed=1, density=100, blur=False, shady=False):
    drops = []
    x_max = imshape[1] if slant < 0 else imshape[1] - slant
    x_min = max(slant, 0) 
    y_max = imshape[0] - drop_length
    y_min = 0
    while True:
        if len(drops) == 0:
            for i in range(density):    
                x = np.random.randint(x_min, x_max)        
                y = np.random.randint(0,y_max)        
                drops.append([x,y]) 
        else:
            for drop in drops:
                drop[0] += slant*speed
                drop[1] += drop_length*speed
                if drop[0] > x_max or drop[0] < x_min or drop[1] > y_max or drop[1] < y_min:
                    if np.random.rand() < 0.5:
                        drop[0] = np.random.randint(x_min, x_max)
                        drop[1] = y_min
                    else:
                        if slant > 0:
                            drop[0] = x_min
                            drop[1] = np.random.randint(0,y_max) 
                        elif slant < 0:
                            drop[0] = x_max
                            drop[1] = np.random.randint(0,y_max) 
                        else:
                            drop[0] = np.random.randint(x_min, x_max)
                            drop[1] = y_min       
        int_drops = np.round(drops).astype(np.uint16)
        image = generate_rain(drops, imshape, slant, drop_length, drop_width, drop_color, blur, shady)
        yield image


def generate_rain(drops, imshape=(64, 64, 3), slant=1, drop_length=2, drop_width=1, drop_color=(255, 255, 255), blur=True, shady=False):
    image = np.zeros(imshape, np.uint8)
    for rain_drop in drops:        
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)    
    if blur:
        image = cv2.blur(image,(3,3))        
    if shady:
        brightness_coefficient = 0.7     
    else:
        brightness_coefficient = 1
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    return image_RGB


def gen_videos(dest_path, num_videos, num_frames=1000):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with trange(num_videos) as t:
        t.set_description('Generating rain videos')
        for i in t:
            # create a rain video with a random setting
            slant = np.random.randint(SLANT_MIN, SLANT_MAX)
            drop_len = np.random.randint(LENGTH_MIN, LENGTH_MAX)
            drop_width = np.random.randint(WIDTH_MIN, WIDTH_MAX)
            speed = np.random.randint(SPEED_MIN, SPEED_MAX)
            density = np.random.randint(DENSITY_MIN, DENSITY_MAX)
#             speed = SPEED
#             density = DENSITY
            video_gen = generate(slant=slant, drop_length=drop_len, drop_width=drop_width, speed=speed, density=density)
            images_arr = []
            for _ in range(num_frames):
                images_arr.append(next(video_gen))
            video_name = "{}_{}_{}_{}_{}.avi".format(slant, drop_len, drop_width, speed, density)
            out = cv2.VideoWriter(os.path.join(dest_path, video_name), cv2.VideoWriter_fourcc(*'DIVX'), 10, (64, 64))
            for img in images_arr:
                out.write(img)
            out.release()


train_path = './videos/rain_train/'
test_path = './videos/rain_test/'

gen_videos(train_path, 200, 2000)
gen_videos(test_path, 200, 2000)