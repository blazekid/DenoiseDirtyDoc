import concurrent.futures
import csv
import logging
import random
import os
import cv2


import joblib
import numpy as np
import sklearn.ensemble
import sklearn.cross_validation
import sklearn.metrics

import skimage.data

from pathlib import Path

from PIL import Image as image

TRAIN_DIR = Path("/home/sandeep/Desktop/Major/RandomForests/input")
TARGET_DIR = Path("/home/sandeep/Desktop/Major/RandomForests/correct")
TEST_DIR = Path("/home/sandeep/Desktop/Major/RandomForests/input")

CHUNKSIZE = 1000000

logging.basicConfig(level=logging.INFO)

def write_image(img, path):
    return cv2.imwrite(path, img)


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


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
        x = image_matrix(load_image(os.path.join('/home/sandeep/Desktop/Major/RandomForests/input', fname)))
        y = image_matrix(load_image(os.path.join('/home/sandeep/Desktop/Major/RandomForests/correct', fname)))
        for i in range(0, x.shape[0]):
            xs.append(x[i, :, :, :].reshape((1, 1, 258, 540)))
            ys.append(y[i, :, :, :].reshape((1, 1, 258, 540)))
    return np.vstack(xs), np.vstack(ys)


def load_test_file(fname, folder):
    xs = []
    x = image_matrix(load_image(os.path.join(folder, fname)))
    for i in range(0, x.shape[0]):
        xs.append(x[i, :, :, :].reshape((1, 1, 258, 540)))
    return np.vstack(xs)


def get_padded(imgarray, padding=1):
    padval = int(round(imgarray.flatten().mean()))
    rows, cols = imgarray.shape
    xpad = np.full((rows, padding), padval, dtype='uint8')
    ypad = np.full((padding, cols + 2 * padding), padval, dtype='uint8')
    return np.vstack((ypad, np.hstack((xpad, imgarray, xpad)), ypad))


def get_features_for_image(imgarray, padding=1):
    rows, cols = imgarray.shape
    padded = get_padded(imgarray, padding=padding)
    features = []
    return np.vstack(tuple(
        np.vstack(tuple(
            padded[i: i + 2 * padding + 1, j: j + 2 * padding + 1].reshape((1, -1))
            for j in range(cols)
        )) for i in range(rows)
    ))


def get_features_for_path(path, padding=1):
    return get_features_for_image(skimage.data.imread(str(path)), padding=padding)


def get_target_for_path(path):
    return skimage.data.imread(str(path)).flatten() / 255


def get_training_sets():
    X = list(joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_features_for_path)(i)
        for i in TRAIN_DIR.iterdir()))
    y = list(joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_target_for_path)(i)
        for i in TARGET_DIR.iterdir()))
    X = np.concatenate(X)
    y = np.concatenate(y)
    logging.info("Finished loading")
    return X, y


def get_model(X, y):
    model = sklearn.ensemble.RandomForestRegressor(
        n_estimators=0, warm_start=True, n_jobs=-1)
    indices = list(range(0, X.shape[0], CHUNKSIZE))
    indices.append(X.shape[0])
    for i in range(len(indices) - 1):
        if not (i + 1) % 10:
            logging.info("Fitting {} of {}".format(i + 1, len(indices) - 1))
        start, end = indices[i], indices[i + 1]
        model.set_params(n_estimators=model.get_params()["n_estimators"] + 1)
        model.fit(X[start: end], y[start: end])
    logging.info("Finished Training")
    return model


def get_index_and_features(path):
    imgarray = skimage.data.imread(str(path))
    X = get_features_for_image(imgarray)
    index = []
    for i in range(imgarray.shape[0]):
        for j in range(imgarray.shape[1]):
            index.append("{}_{}_{}".format(path.stem, i + 1, j + 1))
    return index, X

def list_images(folder):
    included_extentions = ['jpg','bmp','png','gif' ]
    results = [fn for fn in os.listdir(folder) if any([fn.endswith(ext) for ext in included_extentions])]
    return results


def do_test(inFolder, outFolder, nn):
    test_images = list_images(inFolder)
    nTest = len(test_images)
    for x in range(0, nTest):
        fname = test_images[x]
        x1 = list(joblib.Parallel(n_jobs=-1)(joblib.delayed(get_features_for_path)(os.path.join(inFolder, fname))))
        #image_matrix(load_image(os.path.join(inFolder, fname))).load_test_file(fname, inFolder)
        x1 = np.concatenate(x1)
        x1 = x1 - 0.5
        pred_y = nn.predict(x1)
        tempImg = []
        if pred_y.shape[0] == 1:
            tempImg = pred_y[0, 0, :, :].reshape(258, 540)
        if pred_y.shape[0] == 2:
            tempImg1 = pred_y[0, 0, :, :].reshape(258, 540)
        tempImg2 = pred_y[1, 0, :, :].reshape(258, 540)
        tempImg = np.empty((420, 540))
        tempImg[0:258, 0:540] = tempImg1
        tempImg[162:420, 0:540] = tempImg2
        tempImg[tempImg < 0] = 0
        tempImg[tempImg > 1] = 1
        tempImg = np.asarray(tempImg*255.0, dtype=np.uint8)
        write_image(tempImg, (os.path.join(outFolder, fname)))


def write_submission(model, index, X, path):
    with path.open('w', encoding='utf-8', newline='') as outf:
        writer = csv.writer(outf)
        #writer.writerow(('id', 'value'))
        writer.writerows(zip(index, model.predict(X)))

def get_test_set():
    index = []
    X = []
    for imgindex, imgfeatures in joblib.Parallel(n_jobs=-1)(
            joblib.delayed(get_index_and_features)(i)
            for i in TEST_DIR.iterdir()):
        index.extend(imgindex)
        X.append(imgfeatures)
    logging.info("Finished Loading Test Set")
    X = np.vstack(X)
    assert(len(index) == X.shape[0])
    return index, X
  

def main():
    trainX, trainy = get_training_sets()
    model = get_model(trainX, trainy)
    index, testX = get_test_set()
    Y_test = model.predict(testX)
    #print "Accuracy Is " % (abs(Y_test - trainy).mean())
    print "The accuracy of this network is: %0.2f" % (abs(Y_test - trainy)).mean()
    #do_test("/home/sandeep/Desktop/train", '/home/sandeep/Desktop/output', model)
    #write_submission(model, index, testX, Path("submission.csv"))
    
#if __name__ == "__main__":    
main()
