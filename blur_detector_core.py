# import the necessary packages
import numpy as np
from scipy.ndimage import variance
from skimage.filters import laplace

def compute_fft(image, size=60):
	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

	# compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	# zero-out the center of the FFT shift (i.e., remove low
	# frequencies), apply the inverse shift such that the DC
	# component once again becomes the top-left, and then apply
	# the inverse FFT
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

	# compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)

	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
	return mean

def compute_laplace_var(image, k = 3):
	return variance(laplace(image, ksize=k))

def compute_laplace_max(image, k = 3):
	return np.amax(laplace(image, ksize=k))

def compute_laplace_var_grid(image, gridsize = 8, percentile_mean = 80):
	img_width=image.shape[1]
	img_height=image.shape[0]
	
	y1 = 0
	M = img_width//gridsize
	N = img_height//gridsize
	
	laplace_var_blocks = []

	for x in range(0,img_width,M):
		for y in range(0, img_height, N):
			x1 = x + M
			y1 = y + N
			block = image[x:x+M,y:y+N]
			laplace_var_blocks.append(variance(laplace(block)))
	
	# trim 'nan' from the list
	laplace_var_blocks = [x for x in laplace_var_blocks if str(x) != 'nan']
	
	# compute top percentile mean of laplace variance
	return np.mean([x for x in laplace_var_blocks if x >= np.percentile(laplace_var_blocks, percentile_mean)])