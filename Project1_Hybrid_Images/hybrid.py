import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    def grey_correlation(img,kernel):
        imgHeight = len(img)
        imgWidth = len(img[0])
        kernelHeight = len(kernel)
        kernelwidth = len(kernel[0])
        heightIncrease = int((kernelHeight-1)/2)
        widthIncrease = int((kernelwidth-1)/2)
        new = np.zeros((imgHeight,imgWidth),dtype = type(img[0,0]))
        # fullfill the original image
        img = np.append(np.array([[0 for i in range(imgWidth)] for i in range(heightIncrease)]),img,axis = 0)
        img = np.append(img,np.array([[0 for i in range(imgWidth)] for i in range(heightIncrease)]),axis = 0)
        img = np.concatenate((img,np.array([[0 for i in range(widthIncrease)] for i in range((imgHeight+heightIncrease*2))])),axis = 1)
        img = np.concatenate((np.array([[0 for i in range(widthIncrease)] for i in range((imgHeight+heightIncrease*2))]),img),axis = 1)
        for i in range(widthIncrease,imgWidth+widthIncrease):
            for j in range(heightIncrease,imgHeight+heightIncrease):
                new[j-heightIncrease,i-widthIncrease] = (img[j-heightIncrease:j+heightIncrease+1,
                	i-widthIncrease:i+widthIncrease+1]*kernel).sum()
        return new
    if len(img.shape) == 3:
        imgr = grey_correlation(img[:,:,0],kernel)
        imgg = grey_correlation(img[:,:,1],kernel)
        imgb = grey_correlation(img[:,:,2],kernel)
        return np.dstack((imgr,imgg,imgb))
    else:
        return grey_correlation(img,kernel)

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    def grey_convolve(img,kernel):
        imgHeight = len(img)
        imgWidth = len(img[0])
        kernelHeight = len(kernel)
        kernelwidth = len(kernel[0])
        heightIncrease = int((kernelHeight-1)/2)
        widthIncrease = int((kernelwidth-1)/2)
        kernel = np.flip(kernel,axis = 0)
        kernel = np.flip(kernel,axis = 1)
        new = np.zeros((imgHeight,imgWidth),dtype = type(img[0,0]))
        # fullfill the original image
        img = np.append(np.array([[0 for i in range(imgWidth)] for i in range(heightIncrease)]),img,axis = 0)
        img = np.append(img,np.array([[0 for i in range(imgWidth)] for i in range(heightIncrease)]),axis = 0)
        img = np.concatenate((img,np.array([[0 for i in range(widthIncrease)] for i in range((imgHeight+heightIncrease*2))])),axis = 1)
        img = np.concatenate((np.array([[0 for i in range(widthIncrease)] for i in range((imgHeight+heightIncrease*2))]),img),axis = 1)
        for i in range(widthIncrease,imgWidth+widthIncrease):
            for j in range(heightIncrease,imgHeight+heightIncrease):
                new[j-heightIncrease,i-widthIncrease] = (img[j-heightIncrease:j+heightIncrease+1,
                	i-widthIncrease:i+widthIncrease+1]*kernel).sum()
        return new
    if len(img.shape) == 3:
        imgr = grey_convolve(img[:,:,0],kernel)
        imgg = grey_convolve(img[:,:,1],kernel)
        imgb = grey_convolve(img[:,:,2],kernel)
        return np.dstack((imgr,imgg,imgb))
    else:
        return grey_convolve(img,kernel)
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    gaussian = np.zeros((height,width),dtype = np.float64)
    for i in range(-(width-1)/2,(width-1)/2+1):
        for j in range(-(height-1)/2,(height-1)/2+1):
            gaussian[j+(height-1)/2,i+(width-1)/2] = 1/(2*np.pi*sigma**2)*np.exp(-(i**2+j**2)/np.float64((2*sigma**2)))
    return gaussian/gaussian.sum()
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma,size,size)
    return convolve_2d(img,kernel)
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    detail = img - low_pass(img,sigma,size)
    return detail
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


