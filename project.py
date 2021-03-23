from matplotlib import pyplot as plt
from cv2 import cv2
import numpy as np


# read image as it is
def read_img(file_name):
	img = cv2.imread(file_name)
	return img


# resize image with fixed aspect ratio
def resize_img(image, scale):
	res = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
	return res


# convert image to grayscale
def img_to_gray(image):
	img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	return img_gray


# median blur
def median_blur(image):
	blurred_img = cv2.medianBlur(image, 3)
	return blurred_img



def adaptive_thresh(image):
	img_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
	# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dsta
	return img_thresh






# calculate scale and fit into display
def display(window_name, image):
	screen_res = 960, 540	
	
	scale_width = screen_res[0] / image.shape[1]
	scale_height = screen_res[1] / image.shape[0]
	scale = min(scale_width, scale_height)
	window_width = int(image.shape[1] * scale)
	window_height = int(image.shape[0] * scale)

	# reescale the resolution of the window
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window_name, window_width, window_height)

	# display image
	cv2.imshow(window_name, image)
	# wait for any key to quit the program
	cv2.waitKey(0)
	cv2.destroyAllWindows()


max_val = 8
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()

test_img = read_img('files/Test/1000_dollar.jpg')

# resizing must be dynamic
original2 = resize_img(test_img, 0.4)
display('Input Image', original2)
original1 = img_to_gray(original2)
original3 = median_blur(original1)
original = adaptive_thresh(original3)
display('Input Processed Image', original)
# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = [
	'files/Train/india_10_1.jpg', 
	'files/Train/india_10_2.jpg', 
	'files/Train/india_20.jpg', 
	'files/Train/india_50.jpg', 
	'files/Train/india_100_1.jpg', 
	'files/Train/india_100_2.jpg', 
	'files/Train/india_200.jpg', 
	'files/Train/india_500.jpg', 
	'files/Train/india_2000.jpg', 
	'files/Train/dollar_1.jpg',
	'files/Train/dollar_2.jpg',
	'files/Train/dollar_50.jpg',
	'files/Train/dollar_100.jpg',
	'files/Train/pakistan_20.jpg', 
	'files/Train/pakistan_50.jpg', 
	'files/Train/pakistan_100.jpg', 
	'files/Train/pakistan_1000.jpg', 
	'files/Train/russian_10.jpg',
	'files/Train/russian_50.jpg',
	'files/Train/russian_100.jpg',
	'files/Train/russian_1000.jpg',
    'files/Train/singapore_50.jpg',
    'files/Train/singapore_100.jpg',
    'files/Train/singapore_1000.jpg'
	  ]


for i in range(0, len(training_set)):
	# train image
	train_img1 = cv2.imread(training_set[i])
	train_img = cv2.cvtColor(train_img1, cv2.COLOR_BGR2GRAY) 

	(kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img = cv2.imread(training_set[max_pt])
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[12:-4]
	print('\nDetected note: ', note)
	(plt.imshow(img3), plt.show())


else:
	print('No Matches')