import cv2
import numpy as np


IMG1 = "filter1_img.jpg"
IMG2 = "filter2_img.jpg"
GAUSSIAN_3x3 = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16
GAUSSIAN_5x5 = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]) / 273
DOG_GX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
DOG_GY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])


def main():
   for img in IMG1, IMG2:
      perform_gaussian(img)
      perform_dog(img)


def perform_dog(img_name):
   """
   function to load an image, apply padding of width 1, apply derivate of gaussian filters
   to an image, use them to create a sobel filtered image, then save the filtered images
   """
   #load image
   img = load_image(img_name) 

   #zero pad width 1, apply 3x3 derivative of gaussian gx and gy filters
   img_pad = pad_image(img, pad_width=1)
   img_dog_gx = apply_filter(img_pad, filter=DOG_GX)
   img_dog_gy = apply_filter(img_pad, filter=DOG_GY)

   #use gx and gy to compute the Sobel filtered image
   img_sobel = np.sqrt(img_dog_gx.astype(np.float32)**2 + img_dog_gy.astype(np.float32)**2)

   #display filtered images
   """
   display_image(img_dog_gx, img_name, "DoG-gx")
   display_image(img_dog_gy, img_name, "DoG-gy")
   display_image(img_sobel, img_name, "Sobel")
   """
   #save result images 
   save_image(img_dog_gx, img_name, "DoG-gx")
   save_image(img_dog_gy, img_name, "DoG-gy")
   save_image(img_sobel, img_name, "Sobel")


def perform_gaussian(img_name):
   """
   function to load an image, apply a 3x3 gaussian filter with zero-padding of width 1,
   apply a 5x5 gaussian filter with zero-padding of width 2, then save the resulting images
   """
   #load image
   img = load_image(img_name) 

   #zero pad width 1, apply 3x3 gaussian filter
   img_pad = pad_image(img, pad_width=1)
   img_gauss3x3 = apply_filter(img_pad, filter=GAUSSIAN_3x3)

   #zero pad width 2, apply 5x5 gaussian filter
   img_pad = pad_image(img, pad_width=2)
   img_gauss5x5 = apply_filter(img_pad, filter=GAUSSIAN_5x5)

   #display filtered images
   """
   display_image(img_gauss3x3, img_name, "3x3Gaussian")
   display_image(img_gauss5x5, img_name, "5x5Gaussian")
   """
   #save result images 
   save_image(img_gauss3x3, img_name, "3x3Gaussian")
   save_image(img_gauss5x5, img_name, "5x5Gaussian")


def apply_filter(img, filter):
   """
   applies a given filter to the input image.  assumes any desired padding is done prior
   to this function, and also assumes a square filter
   """
   filter_size = len(filter[0])
   height, width = img.shape
   result = []
   for row in range(height - filter_size + 1):
      result.append([])
      for column in range(width - filter_size + 1):
         value = np.sum(img[row:row+filter_size, column:column+filter_size] * filter)
         result[row].append(value)

   return np.array(result, dtype=np.uint8) 


def pad_image(img, pad_width):
   """
   applies zero-padding of pad_width width to img
   """
   img_padded = np.pad(img, pad_width, mode='constant', constant_values=0)
   return img_padded


def load_image(filename):
   """
   loads image as grayscale from file
   """
   img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
   return img


def display_image(img, img_name, msg):
   """
   normalizes pixel values, then displays image
   """
   cv2.imshow(img_name + " " + msg, img)
   img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
   cv2.waitKey(0)


def save_image(img, img_name, msg):
   """
   normalizes pixel values, then saves image to file
   """
   name = img_name.removesuffix(".jpg")
   img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
   cv2.imwrite(name + "-" + msg + ".jpg", img)


if __name__ == "__main__":
   main()