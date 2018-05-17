import cv2
import numpy as np


class box:
	def __init__(self):
		self.x1 = 1080
		self.x2 = 0
		self.y1 = 1920
		self.y2 = 0

	def __str__(self):
		return "x1:" + str(self.x1) + " x2:" + str(self.x2) + " y1:" + str(self.y1) + " y2:" + str(self.y2) 

	def makeInt(self):
		self.x1 = int(self.x1)
		self.x2 = int(self.x2)
		self.y1 = int(self.y1)
		self.y2 = int(self.y2)

	def extend(self, percentage  = 40):
		fraction = percentage/100.0
		delX = self.x2 - self.x1
		delY = self.y2 - self.y1
		meanX = (self.x1 + self.x2)/2.0
		meanY = (self.y1 + self.y2)/2.0
		delta = max(delX, delY)
		delta += fraction*delta
		self.x1 = max(meanX - delta/2.0, 0)
		self.x2 = min(meanX + delta/2.0, 1080)
		self.y1 = max(meanY - delta/2.0, 0)
		self.y2 = min(meanY + delta/2.0, 1920)
		self.makeInt()
		return int(delta)



def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img