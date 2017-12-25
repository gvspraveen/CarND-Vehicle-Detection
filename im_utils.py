import cv2
import matplotlib.image as mpimg

def read_cv2_image(path):
    """
    Reads the image from path and converts to rgb space
    :param path:
    :return:
    """
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_mpimg(path):
    return mpimg.imread(path)