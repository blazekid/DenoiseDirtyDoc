import numpy as np
from scipy import signal, ndimage
from PIL import Image
import os
import cv2

def list_images(folder):
    included_extentions = ['jpg','bmp','png','gif' ]
    results = [fn for fn in os.listdir(folder) if any([fn.endswith(ext) for ext in included_extentions])]
    return results

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def write_image(img, path):
    return cv2.imwrite(path, img)

def image_matrix(img):
    if img.shape[0] == 258:
        return (img[0:258, 0:540] / 255.0).astype('float32').reshape((1, 1, 258, 540))
    if img.shape[0] == 420:
        result = []
        result.append((img[0:258, 0:540] / 255.0).astype('float32').reshape((1, 1, 258, 540)))
        result.append((img[162:420, 0:540] / 255.0).astype('float32').reshape((1, 1, 258, 540)))
        result = np.vstack(result).astype('float32').reshape((2, 1, 258, 540))
    return result

def load_train_set(file_list):
    xs = []
    ys = []
    for fname in file_list:
        x = image_matrix(load_image(os.path.join('/home/sandeep/Desktop/Major/BgRemoval/correct', fname)))
        y = image_matrix(load_image(os.path.join('/home/sandeep/Desktop/Major/BgRemoval/generated', fname)))
        for i in range(0, x.shape[0]):
            xs.append(x[i, :, :, :].reshape((1, 1, 258, 540)))
            ys.append(y[i, :, :, :].reshape((1, 1, 258, 540)))
    return np.vstack(xs), np.vstack(ys)

def load_im(path):
    return np.asarray(Image.open(path))/255.0

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def denoise_im_with_back(inp):
    bg = signal.medfilt2d(inp, 11)
    save('background.png', bg)
    mask = inp < bg - 0.1    
    save('foreground_mask.png', mask)
    back = np.average(bg)
    
    mod = ndimage.filters.median_filter(mask,2)
    mod = ndimage.grey_closing(mod, size=(2,2))
          
    out = np.where(mod, inp, back)    
    return out

for f in os.listdir("input/"):
	#save('/home/sandeep/Desktop/Major/Highpassfilter/Results/'+f,newimage);
	inp_path = '/home/sandeep/Desktop/Major/BgRemoval/input/'+f
	out_path = '/home/sandeep/Desktop/Major/BgRemoval/generated/'+f
	inp = load_im(inp_path)
	out = denoise_im_with_back(inp)
	save(out_path, out)

TRAIN_IMAGES = list_images("/home/sandeep/Desktop/Major/BgRemoval/correct/")   
train_x, train_y = load_train_set(TRAIN_IMAGES)
test_x = train_x
test_y = train_y

print "The accuracy of this network is: %0.2f" % (abs(test_x - test_y)).mean()



	

#inp_path = 'dirty2.png'
#out_path = 'output.png'

#inp = load_im(inp_path)
#out = denoise_im_with_back(inp)

#save(out_path, out)
