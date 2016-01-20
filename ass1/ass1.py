import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# opencv stores the pixels as blue, green and red
BLUE = 0
GREEN = 1
RED = 2
GRAY = 0
# p is the probability parameter used for adding salt pepper noise
p = 0.1
# Mean Kernel
mean_kernel = np.array([[1.0/9.0, 1.0/9.0, 1.0/9.0],[1.0/9.0, 1.0/9.0, 1.0/9.0],[1.0/9.0, 1.0/9.0, 1.0/9.0]])
# Sobel Kernels
sobel_x_kernel = np.array([[-1.0, 0.0, 1.0],[-2.0, 0.0, 2.0],[-1.0, 0.0, 1.0]])
sobel_y_kernel = np.array([[-1.0, -2.0, -1.0],[0.0, 0.0, 0.0],[1.0, 2.0, 1.0]])
# Gaussian Kernel with sigma 1
gauss_kernel = np.array([[1.0/16.0, 1.0/8.0, 1.0/16.0],[1.0/8.0, 1.0/4.0, 1.0/8.0],[1.0/16.0, 1.0/8.0, 1.0/16.0]])
# gauss_kernel = np.array([
# 					[0.0029690, 0.0133062, 0.0219382, 0.0133062, 0.0029690],
# 					[0.0133062, 0.0596343, 0.0983203, 0.0596343, 0.0133062],
# 					[0.0219382, 0.0983203, 0.1621028, 0.0983203, 0.0219382],
# 					[0.0133062, 0.0596343, 0.0983203, 0.0596343, 0.0133062],
# 					[0.0029690, 0.0133062, 0.0219382, 0.0133062, 0.0029690]
# 				])

def display_and_wait(img, window_name="Image"):
	cv2.imshow(window_name, img)
	cv2.waitKey(0)

def display(img, window_name="Image"):
	cv2.imshow(window_name, img)

def save(img, name):
	cv2.imwrite(name, img)

def wait():
	c = 0
	while c != 27:
		c = cv2.waitKey(0)

def grayscale(img):
	b, g, r = img[:,:,0],img[:,:,1],img[:,:,2]
	gray_image = 0.299*r + 0.587*g + 0.114*b

	# display_and_wait(gray_image)
	return gray_image.astype(np.uint8)

def saltpepper_noise(img, p):
	# we assume that the image is grayscale here
	# for each pixel we will color the pixel black of white based on the probability p
	
	# First create a blank image with all zeros
	salt_pepper_image = np.zeros(img.shape, np.uint8)

	height, width = img.shape
	for i in range(height):
		for j in range(width):
			rand = np.random.random_sample()
			if rand < p:
				# The pixel is chosen to be colored white or black
				if np.random.random_sample() < 0.5:
					salt_pepper_image.itemset((i, j), 0)
				else:
					salt_pepper_image.itemset((i, j), 255)
			else:
				# copy the pixel from original image
				salt_pepper_image.itemset((i, j), img.item((i, j)))

	# display_and_wait(salt_pepper_image, "salt pepper image")
	return salt_pepper_image

def pad_image(img, padding):
	# Create a blank image with all zeros of size of the original image + padding
	padded_image = np.zeros(((img.shape[0] + 2 * padding), (img.shape[1] + 2 * padding)), np.uint8)
	# copy the original image to padded image
	height, width = img.shape
	for i in range(height):
		for j in range(width):
			padded_image.itemset((i + padding, j + padding), img.item(i, j))

	return padded_image

def custom_convolute(img, matrix):
	# among parameters first is the image and second is the convolution matrix
	# assumptions that matrix will be having odd number of rows and colums

	# fist pad the image with zeros
	padding = matrix.shape[0] >> 1
	padded_image = pad_image(img, padding)
	convoluted_image = np.zeros(img.shape, np.uint8)

	height, width = img.shape
	# display_and_wait(padded_image, "padded image")
	# Now perform the convolution operation
	for i in range(height):
		for j in range(width):
			conv_sum = 0.0
			for x in range(-padding, padding+1):
				for y in range(-padding, padding+1):
					conv_sum += padded_image.item((i + padding + x), (j + padding + y)) * matrix.item((x + padding), (y + padding))
			
			convoluted_image.itemset((i, j), conv_sum)

	# display_and_wait(convoluted_image, "convoluted Image")
	return convoluted_image

def convolute_fft(img, matrix):
	# convoloveimg=(convolveimg-np.amin(convolveimg)/(np.amax(convolveimg)-np.amin(convolveimg))
	return np.fft.irfft2(np.fft.rfft2(matrix) * np.fft.rfft2(img, matrix.shape))

def fft_convolve2d(x,y):
	return np.fft.irfft2(np.fft.rfft2(x) * np.fft.rfft2(y, x.shape))

def convolute(img, matrix):
	return custom_convolute(img, matrix)
	# return convolute_fft(img, matrix)

def median_filter(img):
	# fist pad the image with zeros
	padding = 1
	padded_image = pad_image(img, padding)
	# Create a blank image to store the median filter
	median_image = np.zeros(img.shape, np.uint8)
	# Now iterate on each pixel
	height, width = img.shape
	for i in range(height):
		for j in range(width):
			# 3 x 9 dimensional array
			medians = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
			for x in range(-padding, padding+1):
				for y in range(-padding, padding+1):
						medians.itemset(((x + padding) * 3 + (y + padding)), padded_image.item((i + padding + x), (j + padding + y)))
			median_image.itemset((i, j), np.median(medians))

	# display_and_wait(median_image, "median Image")
	return median_image

def MSE(img1, img2):
	# Now iterate on each pixel
	height, width = img1.shape
	mse = 0.0
	for i in range(height):
		for j in range(width):
			mse += (img1.item((i, j)) - img2.item((i, j))) ** 2
	mse = mse / width / height
	return mse

def psnr(img1, img2):
	mse = MSE(img1, img2)
	return 10 * np.log10((255.0 ** 2) / mse)

def sobel_filter(img):
	sobel_x_grad_image = img.copy().astype(float)
	sobel_y_grad_image = img.copy().astype(float)
	gradient_mag_image = img.copy().astype(float)
	edges_image = img.copy().astype(float)

	WINDOWSIZE = 1
	height, width = img.shape
	for i in range(1, height-1):
		for j in range(1, width-1):
			sum_x = 0
			sum_y = 0
			for x in range(-WINDOWSIZE, +WINDOWSIZE+1):
				for y in range(-WINDOWSIZE, +WINDOWSIZE+1):
					pixel = img.item(i + x ,j + y)
					sum_x += pixel * sobel_x_kernel[x + WINDOWSIZE][y + WINDOWSIZE]
					sum_y += pixel * sobel_y_kernel[x + WINDOWSIZE][y + WINDOWSIZE]

			gradient_mag_image.itemset((i,j),((sum_x**2 + sum_y**2)**0.5))
			sobel_x_grad_image.itemset((i,j), sum_x)
			sobel_y_grad_image.itemset((i,j), sum_y)

	# Normalize images	
	sobel_x_grad_image = (sobel_x_grad_image - np.amin(sobel_x_grad_image)) * 255 / (np.amax(sobel_x_grad_image) - np.amin(sobel_x_grad_image))
	sobel_y_grad_image = (sobel_y_grad_image - np.amin(sobel_y_grad_image)) * 255 / (np.amax(sobel_y_grad_image) - np.amin(sobel_y_grad_image))
	gradient_mag_image = (gradient_mag_image - np.amin(gradient_mag_image)) * 255 / (np.amax(gradient_mag_image) - np.amin(gradient_mag_image))

	SOBEL_THRESHOLD = 30
	edges_image = np.greater(gradient_mag_image, SOBEL_THRESHOLD).astype(np.uint8)*255

	sobel_x_grad_image = sobel_x_grad_image.astype(np.uint8)
	sobel_y_grad_image = sobel_y_grad_image.astype(np.uint8)
	gradient_mag_image = gradient_mag_image.astype(np.uint8)

	# display(sobel_x_grad_image, "X gradient")
	# display(sobel_y_grad_image, "Y gradient")
	save(sobel_x_grad_image, "X gradient.png")
	save(sobel_y_grad_image, "Y gradient.png")
	save(gradient_mag_image, "Gradient.png")

	return sobel_x_grad_image, sobel_y_grad_image, gradient_mag_image, edges_image

def opencv_sobel_filter(img):
	sobel_x_grad_image = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
	sobel_y_grad_image = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
	gradient_mag_image = (sobel_x_grad_image**2 + sobel_y_grad_image**2)**0.5

	# Normalize images	
	sobel_x_grad_image = (sobel_x_grad_image - np.amin(sobel_x_grad_image)) * 255 / (np.amax(sobel_x_grad_image) - np.amin(sobel_x_grad_image))
	sobel_y_grad_image = (sobel_y_grad_image - np.amin(sobel_y_grad_image)) * 255 / (np.amax(sobel_y_grad_image) - np.amin(sobel_y_grad_image))
	gradient_mag_image = (gradient_mag_image - np.amin(gradient_mag_image)) * 255 / (np.amax(gradient_mag_image) - np.amin(gradient_mag_image))

	SOBEL_THRESHOLD = 30
	edges_image = np.greater(gradient_mag_image, SOBEL_THRESHOLD).astype(np.uint8)*255

	sobel_x_grad_image = sobel_x_grad_image.astype(np.uint8)
	sobel_y_grad_image = sobel_y_grad_image.astype(np.uint8)
	gradient_mag_image = gradient_mag_image.astype(np.uint8)

	return sobel_x_grad_image, sobel_y_grad_image, gradient_mag_image, edges_image	

# reference : http://cs.stanford.edu/people/eroberts/courses/soco/projects/data-compression/lossy/jpeg/dct.htm
def C(x):
	if x == 0:
		return (1.0/(2.0**1.5))
	return 1.0/2.0

def dct(img):
	img = img.astype(float)
	height, width = img.shape
	N = 8
	cosines = np.zeros((N,N), float)
	DCT = np.zeros(img.shape, float)
	for i in range(N):
		for j in range(N):
			cosines.itemset((i, j), math.cos((2.0*i + 1.0) * j * math.pi / (2.0*N)))
	i = 0
	j = 0
	count = 0
	for i in range(0,height, N):
		for j in range(0, width, N):
			# We iterate block by block
			for si in range(i,i+N):
				for sj in range(j,j+N):
					dct_coefficient = 0.0
					for x in range(i,i+N):
						for y in range(j,j+N):
							dct_coefficient += cosines.item(((x - i),(si - i))) * cosines.item(((y - j),(sj - j))) * img.item((x,y))
					dct_coefficient *= C(si - i) * C(sj - j)
					DCT.itemset((si, sj), dct_coefficient) # root(2 * N) = root(16) = 4
	return DCT

# reference: http://www.csie.ntu.edu.tw/~b89044/home/IPR/2DINVERSE.pdf
def idct(dct_matrix):
	height, width = dct_matrix.shape
	N = 8
	cosines = np.zeros((N,N), float)
	iDCT = np.zeros(dct_matrix.shape, float)
	for i in range(N):
		for j in range(N):
			cosines.itemset((i, j), math.cos((2.0*i + 1.0) * j * math.pi / float(2.0*N)))
	i = 0
	j = 0
	cnt = 0
	for i in range(0,height, N):
		for j in range(0, width, N):
			for si in range(i,i+N):
				for sj in range(j,j+N):
					pixel = 0.0
					for x in range(i,i+N):
						for y in range(j,j+N):
							pixel += C(x - i) * C(y - j) * cosines.item(((si - i),(x - i))) * cosines.item(((sj - j),(y - j))) * dct_matrix.item((x,y))
					iDCT.itemset((si, sj), pixel)

	# display(iDCT, "iDCT")
	# Normalizing and converting to integer type
	iDCT = (iDCT - np.amin(iDCT)) / (np.amax(iDCT) - np.amin(iDCT))
	iDCT255 = iDCT*255
	return iDCT255.astype(np.uint8)

# reference : http://www.cse.psu.edu/~rtc12/CSE486/lecture06.pdf
def harris_corner(img):
	global gauss_kernel
	# Ix, Iy, grad_mag, edges = sobel_filter(img)
	Ix = img.copy().astype(float)
	Iy = img.copy().astype(float)
	WINDOWSIZE = 1
	height, width = img.shape
	for i in range(1, height-1):
		for j in range(1, width-1):
			sum_x = 0
			sum_y = 0
			for x in range(-WINDOWSIZE, +WINDOWSIZE+1):
				for y in range(-WINDOWSIZE, +WINDOWSIZE+1):
					pixel = img.item(i + x ,j + y)
					sum_x += pixel * sobel_x_kernel[x + WINDOWSIZE][y + WINDOWSIZE]
					sum_y += pixel * sobel_y_kernel[x + WINDOWSIZE][y + WINDOWSIZE]
			Ix.itemset((i,j), sum_x)
			Iy.itemset((i,j), sum_y)
	print "Sobel Done"
	height, width = img.shape

	# Create a image which will store the corners
	corners_image = img.copy()
	WINDOWSIZE = gauss_kernel.shape[0] / 2
	# k is empirically determined constant = 0.04 - 0.06
	k = 0.06
	THRESHOLD = 300000000.0

	# Caclulate Ix2, Iy2 and Ixy
	"""
	Ix2 = np.zeros(img.shape, float)
	Iy2 = np.zeros(img.shape, float)
	Ixy = np.zeros(img.shape, float)
	for i in range(height):
		for j in range(width):
			Ix2.itemset((i,j), (Ix.item((i,j)) ** 2))
			Iy2.itemset((i,j), (Iy.item((i,j)) ** 2))
			Ixy.itemset((i,j), (Ix.item((i,j)) * Iy.item((i,j))))
	"""
	Ix2 = Ix**2
	Iy2 = Iy**2
	Ixy = Ix*Iy
	# Convolute with gaussing kernel
	Sx2 = np.zeros(img.shape, float)
	Sy2 = np.zeros(img.shape, float)
	Sxy = np.zeros(img.shape, float)
	for i in range(WINDOWSIZE, height - WINDOWSIZE):
		for j in range(WINDOWSIZE, width - WINDOWSIZE):
			ix2, iy2, ixy = 0, 0, 0
			for x in range(-WINDOWSIZE, WINDOWSIZE+1):
				for y in range(-WINDOWSIZE, WINDOWSIZE+1):
					ix2 += gauss_kernel.item((x + WINDOWSIZE, y + WINDOWSIZE)) * Ix2.item((i + x, j + y))
					iy2 += gauss_kernel.item((x + WINDOWSIZE, y + WINDOWSIZE)) * Iy2.item((i + x, j + y))
					ixy += gauss_kernel.item((x + WINDOWSIZE, y + WINDOWSIZE)) * Ixy.item((i + x, j + y))
			Sx2.itemset((i,j), ix2)
			Sy2.itemset((i,j), iy2)
			Sxy.itemset((i,j), ixy)
	print "Gaussian Done"
	# Compute R scores
	"""
	R = np.zeros(img.shape, float)
	for i in range(1, height-1):
		for j in range(1, width-1):
			# Compute Sx2, Sy2 and Sxy
			sx2 = Sx2.item((i, j))
			sy2 = Sy2.item((i, j))
			sxy = Sxy.item((i, j))
			# H = | Sx2 Sxy |
			#     | Sxy Sy2 |
			# R = Det(H) - k(Trace(H))^2
			det_h = sx2 * sy2 - sxy * sxy
			trace_h = sx2 + sy2
			R.itemset((i, j), det_h - k * (trace_h**2))
	"""
	R = (Sx2 * Sy2 - Sxy**2) - k*((Sx2 + Sy2)**2)

	# Threshold the R values
	WINDOWSIZE = 3
	for i in range(WINDOWSIZE, height - WINDOWSIZE):
		for j in range(WINDOWSIZE, width - WINDOWSIZE):
			r = R.item((i,j))
			if r > THRESHOLD:
				# Check for local maxima
				flag = False
				for x in range(-WINDOWSIZE, WINDOWSIZE+1):
					for y in range(-WINDOWSIZE, WINDOWSIZE+1):
						if x == 0 and y == 0:
							continue
						if r < R.item((i+x, j+y)):
							flag = True
							break
					if flag:
						break
				if not flag:
					corners_image.itemset((i, j), img.item((i, j)))
					cv2.circle(corners_image,(j,i), 10, 255, thickness=1, lineType=8, shift=0)
					print r
	
	# display(corners_image.astype(np.uint8), "Corners")
	save(corners_image, "Corners.png")
	print "Corners Done"
	return corners_image

def opencv_harris_corner(img):
	floating_point_img = np.float32(img)
	R = cv2.cornerHarris(floating_point_img, 2, 3, 0.06)
	R = cv2.dilate(R, None)

	corners_image = img.copy()
	# Threshold the R values
	THRESHOLD = 1000000
	height, width = img.shape
	WINDOWSIZE = 3
	for i in range(WINDOWSIZE, height - WINDOWSIZE):
		for j in range(WINDOWSIZE, width - WINDOWSIZE):
			r = R.item((i,j))
			if r > THRESHOLD:
				# Check for local maxima
				flag = False
				for x in range(-WINDOWSIZE, WINDOWSIZE+1):
					for y in range(-WINDOWSIZE, WINDOWSIZE+1):
						if x == 0 and y == 0:
							continue
						if r < R.item((i+x, j+y)):
							flag = True
							break
					if flag:
						break
				if not flag:
					corners_image.itemset((i, j), img.item((i, j)))
					cv2.circle(corners_image,(j,i), 10, 255, thickness=1, lineType=8, shift=0)
					# print r
	return corners_image	

def opencv_grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def custom_psnr_plot_data(img):
	plot_data = []
	for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
		print p
		salt_pepper_img = saltpepper_noise(img, p)
		mean_filtered_img = convolute(salt_pepper_img, mean_kernel)
		median_filtered_img = median_filter(salt_pepper_img)

		median_psnr = psnr(img, median_filtered_img)
		mean_psnr = psnr(img, mean_filtered_img)

		plot_data.append([p,median_psnr, mean_psnr])
	
	plot_data = np.array(plot_data)
	return plot_data

def opencv_psnr_plot_data(img):
	plot_data = []
	for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
		print p
		salt_pepper_img = saltpepper_noise(img, p)
		median_filtered_img = cv2.medianBlur(salt_pepper_img, 3)
		mean_filtered_img = cv2.blur(salt_pepper_img, (3,3))

		median_psnr = psnr(img, median_filtered_img)
		mean_psnr = psnr(img, mean_filtered_img)

		plot_data.append([p,median_psnr, mean_psnr])
	
	plot_data = np.array(plot_data)
	return plot_data

def print_choices():
	print "Image Processing Assignment 1"
	print "Select your option:"
	print "1.	Display Images"
	print "2.	Grayscale Images"
	print "3.	Salt Pepper Noise and Mean/Median Filtering"
	print "4.	Plot PSNR values"
	print "5.	Sobel Edge"
	print "6.	DCT and IDCT"
	print "7.	Harris Corner"
	print "8.	Destroy All Windows"

def main():
	global p
	global mean_kernel
	# Preprocessing
	print "Preprocessing..."
	# Reading Core Images
	cap_image = cv2.imread("cap.bmp")
	lego_image = cv2.imread("lego.tif")

	# Grayscaling Images
	gray_cap_image = grayscale(cap_image)
	gray_lego_image = grayscale(lego_image)

	while True:
		print_choices()
		choice = raw_input()
		if choice == "1":
			cv2.imshow("Cap", cap_image)
			cv2.imshow("Lego", lego_image)
			print "waiting"
			wait()
			cv2.destroyAllWindows()
		elif choice == "2":
			cv2.imshow("Custom: Grayscale Cap", gray_cap_image)
			cv2.imshow("Custom: Grayscale Lego", gray_lego_image)
			cv2.imshow("OpenCV: Grayscale Cap", opencv_grayscale(cap_image))
			cv2.imshow("OpenCV: Grayscale Lego", opencv_grayscale(lego_image))
			wait()
			cv2.destroyAllWindows()
		elif choice == "3":
			print "Enter the value of probability p in decimal"
			p = float(raw_input())
			salt_pepper_cap_image = saltpepper_noise(gray_cap_image, p)
			cv2.imshow("Cap with salt pepper noise", salt_pepper_cap_image)
			# mean filtering
			mean_filtered_cap_image = convolute(salt_pepper_cap_image, mean_kernel)
			cv2.imshow("Custom: Mean Filtered Cap", mean_filtered_cap_image)
			# median filtering
			median_filtered_cap_image = median_filter(salt_pepper_cap_image)
			cv2.imshow("Custom: Median Filtered Cap", median_filtered_cap_image)
			
			# OpenCV
			# mean filtering
			mean_filtered_cap_image_opencv = cv2.blur(salt_pepper_cap_image, (3,3))
			cv2.imshow("OpenCV: Mean Filtered Cap", mean_filtered_cap_image_opencv)
			# median filtering
			median_filtered_cap_image_opencv = cv2.medianBlur(salt_pepper_cap_image, 3)
			cv2.imshow("OpenCV: Median Filtered Cap", median_filtered_cap_image_opencv)
			wait()
			cv2.destroyAllWindows()
		elif choice == "4":
			# Plot PSNR values
			# Custom
			psnr_plot = custom_psnr_plot_data(gray_cap_image)
			print psnr_plot
			# OpenCV
			psnr_plot_opencv = opencv_psnr_plot_data(gray_cap_image)
			print psnr_plot_opencv

			plt.plot(psnr_plot[:, 0], psnr_plot[:, 1], 'rs')
			plt.plot(psnr_plot[:, 0], psnr_plot[:, 2], 'bs')
			plt.plot(psnr_plot_opencv[:, 0], psnr_plot_opencv[:, 1], 'ro')
			plt.plot(psnr_plot_opencv[:, 0], psnr_plot_opencv[:, 2], 'bo')
			# Adjusting axis for better visibility
			plt.axis([0.09, 0.51, 15,35])
			plt.show()
		elif choice == "5":
			# Sobel	
			# Custom
			sobel_x_image, sobel_y_image, grad_image, edges_image = sobel_filter(gray_lego_image)
			cv2.imshow("Custom: X Gradient", sobel_x_image)
			cv2.imshow("Custom: Y Gradient", sobel_y_image)
			cv2.imshow("Custom: Gradient Magnitude", grad_image)
			cv2.imshow("Custom: Thresholded Edges", edges_image)
			# OpenCV
			sobel_x_image_opencv, sobel_y_image_opencv, grad_image_opencv, edges_image_opencv = opencv_sobel_filter(gray_lego_image)
			cv2.imshow("OpenCV: X Gradient", sobel_x_image_opencv)
			cv2.imshow("OpenCV: Y Gradient", sobel_y_image_opencv)
			cv2.imshow("OpenCV: Gradient Magnitude", grad_image_opencv)
			cv2.imshow("OpenCV: Thresholded Edges", edges_image_opencv)
			wait()
			cv2.destroyAllWindows()
		elif choice == "6":
			# DCT and Inverse DCT
			# Custom
			dct_matrix = dct(gray_cap_image)
			idct_image = idct(dct_matrix)
			cv2.imshow("Custom: DCT", dct_matrix.astype(np.uint8))
			cv2.imshow("Custom: IDCT", idct_image)
			save(idct_image, "inverse.png")
			# OpenCV
			floating_point_img = np.float32(gray_cap_image)/255.0
			dct_matrix_opencv = cv2.dct(floating_point_img)
			idct_image_opencv = cv2.idct(dct_matrix_opencv)
			cv2.imshow("OpenCV: DCT", dct_matrix_opencv.astype(np.uint8))
			cv2.imshow("OpenCV: IDCT", idct_image_opencv)
			wait()
			cv2.destroyAllWindows()
		elif choice == "7":
			# Harris Corner
			# Custom
			harris_corners_image = harris_corner(gray_lego_image)
			cv2.imshow("Custom: Harris Corner", harris_corners_image)
			# OpenCV
			harris_corners_image_opencv = opencv_harris_corner(gray_lego_image)
			cv2.imshow("OpenCV: Harris Corner", harris_corners_image_opencv)
			wait()
			cv2.destroyAllWindows()
		elif choice == "8":
			# Cleanup all windows
			cv2.destroyAllWindows()



if __name__ == '__main__':
	main()
