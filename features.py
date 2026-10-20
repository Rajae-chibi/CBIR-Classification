
import cv2
import numpy as np
from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)


def preprocess_image(img):
    # L'image arrive en RGB, uint8
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return img


def extract_rgb_features(img):
    features = []

    for canal in range(3):
        channel = img[:, :, canal]
        moyenne = np.mean(channel)
        ecart_type = np.std(channel)
        features.extend([moyenne, ecart_type])

    return np.array(features)


def extract_hsv_features(img):
    img_uint8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    hist = hist / (hist.sum() + 1e-8)
    return hist.flatten()


def extract_glcm_features(img):
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "homogeneity",
        "energy",
        "correlation"
    ]

    features = []

    for prop in properties:
        value = graycoprops(glcm, prop)[0, 0]
        features.append(value)

    return np.array(features)


def extract_lbp_features(img):
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    P = 8
    R = 1

    lbp = local_binary_pattern(
        gray,
        P=P,
        R=R,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, P + 3),
        range=(0, P + 2)
    )

    hist = hist.astype("float32")
    hist = hist / (hist.sum() + 1e-8)

    return hist


def extract_hu_features(img):
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu


def extract_all_features(img):
    img = preprocess_image(img)

    features = np.hstack([
        extract_rgb_features(img),
        extract_hsv_features(img),
        extract_glcm_features(img),
        extract_lbp_features(img),
        extract_hu_features(img)
    ])

    return features
