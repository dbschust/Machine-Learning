import cv2
import numpy as np


FRAME1A = "frame1_a.png"
FRAME1B = "frame1_b.png"
FRAME2A = "frame2_a.png"
FRAME2B = "frame2_b.png"
DOG_GX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
DOG_GY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])


def main():
   perform_optical_flow(FRAME1A, FRAME1B)
   perform_optical_flow(FRAME2A, FRAME2B)


def perform_optical_flow(file1, file2):
   """
   wrapper function to to load two images from files, calculate optical flow
   between them, then display visual representation of the optical flow.  
   """
   frame1 = load_image(file1)
   frame2 = load_image(file2)
   Vx, Vy = calc_flow(frame1, frame2)
   display_flow(Vx, Vy, title=f"{file1} to {file2}")
   overlay_flow(Vx, Vy, frame1, title=f"{file1} to {file2}")


def calc_gradients(img):
   """
   pads image with zero padding of width one, then convolves image with
   derivative of Gaussian filters gx and gy.  returns the result as 
   Ix and Iy matrices.
   """
   img_pad = pad_image(img, pad_width=1)
   Ix = apply_filter(img_pad, DOG_GX)
   Iy = apply_filter(img_pad, DOG_GY)
   return Ix, Iy


def apply_filter(img, filter):
   """
   applies a given filter to the input image.  
   assumes any desired padding is done prior to this function.
   """
   fh, fw = filter.shape 
   height, width = img.shape
   result = np.zeros((height - fh + 1, width - fw + 1), dtype=np.float64)

   for row in range(height - fh + 1):
      for column in range(width - fw + 1):
         region = img[row:row+fh, column:column+fw]
         result[row][column] = np.sum(region * filter)

   return np.array(result, dtype=np.float64) 


def calc_flow(frame1, frame2):
   """
   function to calculate optical flow values between two image frames. 
   frame1 and frame2 should be the same dimensions.
   calculates Ix, Iy, and It for all pixels then solves for Vx and Vy of
   each pixel using the window of 9 surrounding pixels.  Returns Vx and Vy
   (same shape as original image) as matrices of the optical flow for the pixels in the image.
   """
   h, w = frame1.shape

   #calculate Ix, Iy, It for all pixels
   It = frame1.astype(np.float64) - frame2.astype(np.float64)
   Ix, Iy = calc_gradients(frame1)
   It = pad_image(It, pad_width=1)
   Ix = pad_image(Ix, pad_width=1)
   Iy = pad_image(Iy, pad_width=1)

   Vx = np.zeros((h, w), dtype=np.float32)
   Vy = np.zeros((h, w), dtype=np.float32)

   for row in range(h):
      for column in range(w):
         #windows of surrounding 9 pixels for Ix, Iy, It
         Ix_win = Ix[row:row+3, column:column+3].flatten()
         Iy_win = Iy[row:row+3, column:column+3].flatten()
         It_win = It[row:row+3, column:column+3].flatten()

         #create matrix A using the Ix and Iy vectors (9x2)
         A = np.stack((Ix_win, Iy_win), axis=1)
         b = -It_win

         #matrix multiplications 
         ATA = A.T @ A
         ATb = A.T @ b

         #avoid division by zero
         if np.linalg.det(ATA) < 1e-2:
            continue

         #solve for v to get Vx and Vy for current pixel, insert result to Vx and Vy matrices
         v = np.linalg.pinv(ATA) @ ATb
         Vx[row, column] = v[0]
         Vy[row, column] = v[1]

   return Vx, Vy 


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


def save_image(img, img_name, msg):
   """
   normalizes pixel values, then saves image to file
   """
   name = img_name.removesuffix(".jpg")
   #img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
   cv2.imwrite(name + "-" + msg + ".jpg", img)


def display_flow(flow_u, flow_v, title="flow"):
   """
   visualizes Vx, Vy, and flow magnitude, saves results to files
   """
   magnitude = np.sqrt(flow_u ** 2 + flow_v ** 2)

   #normalize results to 0–255
   Vx_vis = cv2.normalize(flow_u, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
   Vy_vis = cv2.normalize(flow_v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
   mag_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

   #display results
   #cv2.imshow("Vx (Horizontal Flow)", Vx_vis)
   #cv2.imshow("Vy (Vertical Flow)", Vy_vis)
   #cv2.imshow("Magnitude of Flow", mag_vis)
   #cv2.waitKey(0)
   #cv2.destroyAllWindows()

   #save results
   cv2.imwrite(f"{title}_vx.jpg", Vx_vis)
   cv2.imwrite(f"{title}_vy.jpg", Vy_vis)
   cv2.imwrite(f"{title}_magnitude.jpg", mag_vis)

def overlay_flow(Vx, Vy, img, title, alpha=0.6):
   """
   Superimposes color-coded optical flow vectors onto the grayscale image.
   - Hue encodes direction.
   - Brightness encodes magnitude.
   - Alpha controls blending ratio.
   """

   # Ensure float32 for flow
   Vx = Vx.astype(np.float32)
   Vy = Vy.astype(np.float32)

   # Convert grayscale image to BGR
   gray_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

   # Compute magnitude and angle
   mag, ang = cv2.cartToPolar(Vx, Vy, angleInDegrees=True)

   # Create HSV image
   hsv = np.zeros_like(gray_bgr, dtype=np.uint8)
   hsv[..., 0] = (ang / 2).astype(np.uint8)       # Hue: 0–180
   hsv[..., 1] = 255                              # Saturation
   hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value

   # Convert HSV to BGR
   flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

   # Blend original and flow
   overlay = cv2.addWeighted(gray_bgr, 1 - alpha, flow_bgr, alpha, 0)

   # Display
   #cv2.imshow("Flow Superimposed on Image", overlay)
   #cv2.waitKey(0)
   #cv2.destroyAllWindows()

   # Optional: save result
   cv2.imwrite(f"{title}_flow_overlay.jpg", overlay)
   arrows_img = draw_dense_arrows(img, Vx, Vy)
   cv2.imwrite(f"{title}_arrows.jpg", arrows_img)

import cv2
import numpy as np


def draw_dense_arrows(frame, flow_u, flow_v, step=10):
   """
   Draws dense arrows over an image using Vx (flow_u) and Vy (flow_v).
   `step` controls the sampling rate to avoid cluttering the image.
   """
   h, w = frame.shape
   color_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

   green = (0, 255, 0)  # Bright green in BGR

   for y in range(0, h, step):
      for x in range(0, w, step):
         fx = flow_u[y, x]
         fy = flow_v[y, x]

         # Start and end point of the arrow
         end_x = int(x + fx)
         end_y = int(y + fy)

         cv2.arrowedLine(color_img, (x, y), (end_x, end_y), green, 1, tipLength=0.3)

   return color_img



if __name__ == "__main__":
   main()