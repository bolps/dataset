# import the necessary packages
import cv2
import numpy as np

def compute_fft(image, size=60):
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)
	
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	return mean

def compute_laplace_var(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def compute_laplace_var_grid(image, gridsize = 8, percentile_sharpest = 80):
	height, width = image.shape
	
	gridsize = gridsize
	block_width = width//gridsize
	block_height = height//gridsize
	
	laplace_var_blocks = []
	
	for r in range(0,height,block_height):
		for c in range(0,width,block_width):
			block = image[r:r+block_height, c:c+block_width]
			if (block.shape[0] == block_height and block.shape[1] == block_width):
				laplace_var_block = cv2.Laplacian(block, cv2.CV_64F).var()
				laplace_var_blocks.append(laplace_var_block)
	
	laplace_sharpest_blocks = np.mean([x for x in laplace_var_blocks if x >= np.percentile(laplace_var_blocks, percentile_sharpest)])
	return laplace_sharpest_blocks
