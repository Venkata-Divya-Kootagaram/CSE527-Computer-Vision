
# coding: utf-8

# # CSE527 Homework1
# **Due date: 23:59 on Sep 25, 2018 (Tuesday)**
# 
# ## Prerequisite
# ---
# * **Install Python**: you are strongly recommended to install and use python version of **2.7.x** as some of the packages we use may not support the latest version of Python. you can download it at https://www.python.org/downloads/ or use Anaconda (a Python distribution) at https://docs.continuum.io/anaconda/install/. Below are some materials and tutorials which you may find useful for learning Python if you are new to Python.
#   - https://docs.python.org/2.7/
#   - https://www.learnpython.org/
#   - http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html
#   - http://www.scipy-lectures.org/advanced/image_processing/index.html
# 
# 
# * **Install Python packages**: install Python packages: `numpy`, `matplotlib`, `opencv-python` using pip, for example:
# ```
# pip install numpy matplotlib opencv-python
# ``` 
# Sometimes you may need to use command `pip2` instead of `pip`
# 
# 
# * **Install Jupyter Notebook**: follow the instructions at http://jupyter.org/install.html to install Jupyter Notebook and get yourself familiar with it. *After you have installed Python and Jupyter Notebook, please open the notebook file 'HW1.ipynb' with your Jupyter Notebook and do your homework there.*
# 
# 
# ## Example
# ---
# Please read through the following examples where we apply image thresholding to an image. This example is desinged to help you get familiar with the basics of Python and routines of OpenCV. This part is for your exercises only, you do not need to submit anything from this part.

# In[72]:


# import packages
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display, Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# function for image thresholding
def imThreshold(img, threshold, maxVal):
    assert len(img.shape) == 2 # input image has to be gray
    
    height, width = img.shape
    bi_img = np.zeros((height, width), dtype=np.uint8)
    for x in xrange(height):
        for y in xrange(width):
            if img.item(x, y) > threshold:
                bi_img.itemset((x, y), maxVal)
                
    return bi_img


# In[73]:


# read the image for local directory (same with this .ipynb) 
img = cv2.imread('SourceImages/snow.jpg')

# convert a color image to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# image thresholding using global tresholder
img_bi = imThreshold(img_gray, 127, 255)

# Be sure to convert the color space of the image from
# BGR (Opencv) to RGB (Matplotlib) before you show a 
# color image read from OpenCV
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('original image')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_gray, 'gray')
plt.title('gray image')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_bi, 'gray')
plt.title('binarized image')
plt.axis("off")

plt.show()


# ## Description
# ---
# In this homework, you will work on basic image processing problems with Python2. Make sure to install opencv for your Python2. There are five problems in total with specific instructions for each of them. Be sure to read **Submission Guidelines** below. They are important.

# ## Problems
# ---

# - **Problem 1 Gaussian convolution {20 pts}:** Write a function in Python that takes two arguments, a width parameter and a variance parameter, and returns a 2D array containing a Gaussian kernel of the desired dimension and variance. The peak of the Gaussian should be in the center of the array. Make sure to normalize the kernel such that the sum of all the elements in the array is 1. Use this function and the OpenCV’s `filter2D` routine to convolve the lena and lena_noise arrays with a 5 by 5 Gaussian kernel with sigma of 1. Repeat with a 11 by 11 Gaussian kernel with a sigma of 3. There will be four output images from this problem, namely, lena convolved with 3x3, 11x11, lena_noise convolved with 3x3, and 11x11. Once you fill in and run the codes, the outputs will be saved under Results folder. These images will be graded based on the difference with ground truth images. You might want to try the same thing on other images but it is not required. Include your code and results in your Jupyter Notebook file.

# In[74]:


def genGaussianKernel(width, sigma):
    
    # define you 2d kernel here
    x,y = np.meshgrid(np.arange(-width // 2 + 1., width // 2 + 1.), np.arange(-width // 2 + 1., width // 2 + 1.))
    kernel_2d = np.exp(-(x**2 + y**2) / (2. * sigma**2))
    kernel_2d = kernel_2d/np.sum(kernel_2d)
    return kernel_2d

# Load images
lena       = cv2.imread('SourceImages/lena.bmp', 0)
lena_noise = cv2.imread('SourceImages/lena_noise.bmp', 0)

# Generate Gaussian kernels
kernel_1 = genGaussianKernel(5,1)   # 5 by 5 kernel with sigma of 1
kernel_2 = genGaussianKernel(11,3)  # 11 by 11 kernel with sigma of 3
#print(kernel_1)

# Convolve with lena and lena_noise
res_lena_kernel1 = cv2.filter2D(lena, -1, kernel_1) 
res_lena_kernel2 = cv2.filter2D(lena, -1, kernel_2) 
res_lena_noise_kernel1 = cv2.filter2D(lena_noise, -1, kernel_1) 
res_lena_noise_kernel2 = cv2.filter2D(lena_noise, -1, kernel_2) 

# Write out result images
cv2.imwrite("Results/P1_01.jpg", res_lena_kernel1)
cv2.imwrite("Results/P1_02.jpg", res_lena_kernel2)
cv2.imwrite("Results/P1_03.jpg", res_lena_noise_kernel1)
cv2.imwrite("Results/P1_04.jpg", res_lena_noise_kernel2)

# Plot results
plt.figure(figsize = (10, 10))
plt.subplot(2, 2, 1)
plt.imshow(res_lena_kernel1, 'gray')
plt.title('lena: 5x5 kernel with var as 1')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(res_lena_kernel2, 'gray')
plt.title('lena: 11x11 kernel with var as 3')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(res_lena_noise_kernel1, 'gray')
plt.title('lena_noise: 5x5 kernel with var as 1')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(res_lena_noise_kernel2, 'gray')
plt.title('lena_noise: 11x11 kernel with var as 3')
plt.axis("off")

plt.show()


# - **Problem 2 Separable convolutions {20 pts}:** The Gaussian kernel is separable, which means that convolution with a 2D Gaussian can be accomplished by convolving the image with two 1D Gaussians, one in the x direction and the other one in the y direction. Perform a 11x11 convolution with sigma of 3 from question 1 using this scheme. You can still use `filter2D` to convolve the images with each of the 1D kernel. Verify that you get the similar results with what you did with 2D kernels by computing the difference image between the results from the two methods. This difference image should be close to blank. Include your code and results in your Jupyter Notebook file. There is no output image from this part. Be sure to display the result images in the jupyter file.

# In[77]:


def genGausKernel1D(length, sigma):
    
    # define you 1d kernel here   
    x = np.arange(-length//2 + 1., length//2 + 1.)
    kernel_1d = np.exp(-(x**2) / (2.*sigma**2))
    kernel_1d = kernel_1d/np.sum(kernel_1d)
    return kernel_1d

# Generate two 1d kernels here
width = 11
sigma = 3
kernel_x = genGausKernel1D(width, sigma)
kernel_y = np.reshape(kernel_x, (width,1))

# Generate a 2d 11x11 kernel with sigma of 3 here as before
kernel_2d = genGaussianKernel(width,sigma) 

# Convolve with lena_noise
res_lena_noise_kernel1d_x  = cv2.filter2D(lena_noise, -1, kernel_x)
res_lena_noise_kernel1d_x = cv2.transpose(res_lena_noise_kernel1d_x)
res_lena_noise_kernel1d_xy = cv2.filter2D(res_lena_noise_kernel1d_x, -1, kernel_y) 
res_lena_noise_kernel1d_x = cv2.transpose(res_lena_noise_kernel1d_x)
res_lena_noise_kernel1d_xy = cv2.transpose(res_lena_noise_kernel1d_xy)

res_lena_noise_kernel2d = cv2.filter2D(lena_noise, -1, kernel_2d) 

# Plot results
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.imshow(lena_noise, 'gray')
plt.title('lena_noise')
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(res_lena_noise_kernel1d_x, 'gray')
plt.title('lena_noise with 11x11 GF in X')
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(res_lena_noise_kernel1d_xy, 'gray')
plt.title('lena_noise with 11x11 GF in X and Y')
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(res_lena_noise_kernel2d, 'gray')
plt.title('lena_noise with 11x11 GF in 2D')
plt.axis("off")

plt.show()

# Compute the difference array here
lena_diff = (res_lena_noise_kernel2d-res_lena_noise_kernel1d_xy)

plt.gray()
plt.imshow(lena_diff)


# - **Problem 3 Laplacian of Gaussian {20 pts}:** Convolve an 23 by 23 Gaussian of sigma = 3 with the discrete approximation to the Laplacian kernel [1 1 1; 1 -8 1; 1 1 1]. Plot the Gaussian kernel and 2D Laplacian of Gaussian using the `Matplotlib` function `plot`. Use the `Matplotlib` function `plot_surface` to generate a 3D plot of LoG. Do you see why this is referred to as the Mexican hat filter? Include your code and results in your Jupyter Notebook file. There is no output image from this part.

# In[78]:


width = 23
sigma = 3

# Create your Laplacian kernel
Laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) 

# Create your Gaussian kernel
Gaussian_kernel = genGaussianKernel(width,sigma) 

# Create your Laplacian of Gaussian
LoG = cv2.filter2D(Gaussian_kernel, -1, Laplacian_kernel) 

# Plot Gaussian and Laplacian
fig = plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(Gaussian_kernel, interpolation='none', cmap=cm.jet)
plt.title('Gaussian kernel')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(LoG, interpolation='none', cmap=cm.jet)
plt.title('2D Laplacian of Gaussian')
plt.axis("off")

# Plot the 3D figure of LoG
(X,Y) = np.meshgrid(np.arange(width), np.arange(width))
ax = plt.subplot(1, 3, 3, projection = '3d')
ax.plot_surface(X,Y,LoG)
plt.title('3D Laplacian of Gaussian')

plt.show()


# - **Problem 4 Histogram equalization {20 pts}:** Refer to Szeliski's book on section 3.4.1, and within that section to eqn 3.9 for more information on histogram equalization. Getting the histogram of a grayscale image is incredibly easy with python. A histogram is a vector of numbers. Once you have the histogram, you can get the cumulative distribution function (CDF) from it. Then all you have left is to find the mapping from each value [0,255] to its adjusted value (just using the CDF basically). **DO NOT** use cv2.equalizeHist() directly to solve the exercise! We will expect to see in your code that you get the PDF and CDF, and that you manipulate the pixels directly (avoid a for loop, though). There will be one output image from this part which is the histogram equalized image. It will be compared against the ground truth.

# In[79]:


def histogram_equalization(img_in):
    
    #Converting RGB to YCrCb
    img_y_cr_cb = cv2.cvtColor(img_in, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    
    #Performing histogram equalization on intensity Y
    hist,bins = np.histogram(y.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    transformed = cdf[y]
    merged_image = cv2.merge([transformed, cr, cb])
    img_out = cv2.cvtColor(merged_image, cv2.COLOR_YCrCb2BGR)
    return True, img_out

# Read in input images
img_equal = cv2.imread('SourceImages/hist_equal.jpg', cv2.IMREAD_COLOR)

# Histogram equalization
succeed, output_image = histogram_equalization(img_equal)

# Plot results
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_equal[..., ::-1])
plt.title('original image')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output_image[..., ::-1])
plt.title('histogram equal result')
plt.axis("off")

# Write out results
cv2.imwrite("Results/P4_01.jpg", output_image)


# - **Problem 5 Low and high pass filters {20 pts}:**  follow these tutorials and you should be fine:
# http://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
# http://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html <br>
# For your LPF (low pass filter), mask a 20x20 window of the center of the FT (Fourier Transform) image (the low frequencies). For the HPF, just reverse the mask. The filtered low and high pass images will be the two outputs from this part and automatically saved to Results folder.

# In[80]:


from skimage import img_as_ubyte
def low_pass_filter(img_in):

    # Write low pass filter here
    dft = cv2.dft(np.float32(img_in),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    # mask
    rows, cols = img_in.shape
    crow,ccol = rows/2 , cols/2
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-10:crow+10, ccol-10:ccol+10] = 1
    
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_out = cv2.idft(f_ishift)
    img_out = cv2.magnitude(img_out[:,:,0],img_out[:,:,1])
    img_out.astype(np.uint8)
    return True, img_out


def high_pass_filter(img_in):

    # Write high pass filter here
    f = np.fft.fft2(img_in)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    # mask
    rows, cols = img_in.shape
    crow,ccol = rows/2 , cols/2
    fshift[crow-10:crow+10, ccol-10:ccol+10] = 0
    
    # apply mask and inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_out = np.fft.ifft2(f_ishift)
    img_out = np.abs(img_out)
    return True, img_out

# Read in input images
img_filter = cv2.imread('SourceImages/filter.png', 0)

# Low and high pass filter
succeed1, output_low_pass_image1  = low_pass_filter(img_filter)
succeed2, output_high_pass_image2 = high_pass_filter(img_filter)

# Plot results
fig = plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(img_filter, 'gray')
plt.title('original image')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(output_low_pass_image1, 'gray')
plt.title('low pass result')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(output_high_pass_image2, 'gray')
plt.title('high pass result')
plt.axis("off")

normalizedImg = np.zeros((240, 320))
normalizedImg = cv2.normalize(output_low_pass_image1,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
output_low_pass_image1 = normalizedImg.astype(np.uint8)

cv2.imwrite("Results/P5_01.jpg", output_low_pass_image1)
cv2.imwrite("Results/P5_02.jpg", output_high_pass_image2)


# ## Submission guidelines
# ---
# Extract the downloaded .zip file to a folder of your preference. The input and output paths are predefined and **DO NOT** change them. The image read and write functions are already written for you. All you need to do is to fill in the blanks as indicated to generate proper outputs.
# 
# When submitting your .zip file through blackboard, please <br> 
# -- name your .zip file as Surname_Givenname_SBUID (example: Trump_Donald_11113456). <br>
# -- DO NOT change the folder structre, please just fill in the blanks. <br>
# 
# You are encouraged to make posts and answer questions on Piazza. Due to the amount of emails I receive from past years, it is unfortunate that I won't be able to reply all your emails. Please ask questions on Piazza and send emails only when it is private.
# 
# If you alter the folder strucutres, the grading of your homework will be significantly delayed and possibly penalized. And I **WILL NOT** reply to any email regarding this matter.
# 
# Be aware that your codes will undergo plagiarism checker both vertically and horizontally. Please do your own work.
# 
# Late submission penalty: <br>
# There will be a 10% penalty per day for late submission. However, you will have 3 days throughout the whole semester to submit late without penalty. Note that the grace period is calculated by days instead of hours. If you submit the homework one minute after the deadline, one late day will be counted. Likewise, if you submit one minute after the deadline, the 10% penaly will be imposed if not using the grace period.

# REFERENCES:
# 1. https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
# 2. https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
