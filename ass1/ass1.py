import cv2
import numpy as np

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

def display_and_wait(img, window_name="Image"):
	cv2.imshow(window_name, img)
	cv2.waitKey(0)

def display(img, window_name="Image"):
	cv2.imshow(window_name, img)

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

	height = img.shape[0]
	width = img.shape[1]
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
	height = img.shape[0]
	width = img.shape[1]
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

	height = img.shape[0]
	width = img.shape[1]
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
	height = img.shape[0]
	width = img.shape[1]
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
	height = img1.shape[0]
	width = img1.shape[1]
	mse = 0.0
	for i in range(height):
		for j in range(width):
			mse += (img1.item((i, j)) - img2.item((i, j))) ** 2
	mse = mse / width / height
	return mse

def psnr(img1, img2):
	mse = MSE(img1, img2)
	return 10 * np.log10((255.0 ** 2) / mse)

# def sobel_x_filter(img):


def main():
	global p
	global mean_kernel
	cap_image = cv2.imread("cap.bmp")
	lego_image = cv2.imread("lego.tif")

	# display_and_wait(cap_image, "Cap")
	# display(cap_image, "Cap")
	# display_and_wait(lego_image, "lego")

	gray_cap_image = grayscale(cap_image)
	gray_lego_image = grayscale(lego_image)

	# display_and_wait(gray_cap_image, "gray cap")
	display(gray_cap_image, "gray cap")
	# display_and_wait(gray_lego_image, "gray lego")

	salt_pepper_cap_image = saltpepper_noise(gray_cap_image, p)
	# display_and_wait(salt_pepper_cap_image, "Cap with salt pepper noise")
	display(salt_pepper_cap_image, "Cap with salt pepper noise")
	# mean filtering
	mean_filtered_cap_image = convolute(salt_pepper_cap_image, mean_kernel)
	# display_and_wait(mean_filtered_cap_image, "Mean Filtered Cap")
	display(mean_filtered_cap_image, "Mean Filtered Cap")
	# median filtering
	median_filtered_cap_image = median_filter(salt_pepper_cap_image)
	# display_and_wait(median_filtered_cap_image, "Median Filtered Cap")
	display(median_filtered_cap_image, "Median Filtered Cap")
	
	# TODO: Plot PSNR values
	print psnr(gray_cap_image, mean_filtered_cap_image)
	print psnr(gray_cap_image, median_filtered_cap_image)

	# Sobel x gradient
	# sobel_x_image = sobel_x_filter(gray_lego_image)
	# display_and_wait()
	# sobel_y_image = convolute(gray_lego_image, sobel_y_kernel)

	cv2.waitKey(0)
	# Cleanup all windows
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
