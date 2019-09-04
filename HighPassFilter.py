import os
import cv2
from PIL import Image
import gzip
import numpy as np

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
        x = image_matrix(load_image(os.path.join('/home/sandeep/Desktop/Major/Highpassfilter/correct', fname)))
        y = image_matrix(load_image(os.path.join('/home/sandeep/Desktop/Major/Highpassfilter/generated2', fname)))
        for i in range(0, x.shape[0]):
            xs.append(x[i, :, :, :].reshape((1, 1, 258, 540)))
            ys.append(y[i, :, :, :].reshape((1, 1, 258, 540)))
    return np.vstack(xs), np.vstack(ys)

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)


for f in os.listdir("input2/"):
	imgid = int(f[:-4])
	
	imdata = np.asarray(Image.open("input2/"+f).convert('L'))/255.0
	
	imfft = np.fft.fft2(imdata)
	
	
	for i in range(imfft.shape[0]):
		kx = i/float(imfft.shape[0])
		if kx>0.5: 
			kx = kx-1
			
		for j in range(imfft.shape[1]):
			ky = j/float(imfft.shape[1])
			if ky>0.5: 
				ky = ky-1
				
			if (kx*kx + ky*ky < 0.030*0.030):
				imfft[i,j] = 0
	
	
	newimage = 1.0*((np.fft.ifft2(imfft)).real)+1.0
	
	newimage = np.minimum(newimage, 1.0)
	newimage = np.maximum(newimage, 0.0)

	save('/home/sandeep/Desktop/Major/Highpassfilter/generated2/'+f,newimage)


#TRAIN_IMAGES = list_images("/home/sandeep/Desktop/Major/Highpassfilter/correct/")   
#train_x, train_y = load_train_set(TRAIN_IMAGES)
#test_x = train_x
#test_y = train_y

#print "The accuracy of this network is: %0.2f" % (abs(test_x - test_y)).mean()


