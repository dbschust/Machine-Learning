import cv2


def main():
   #load images, create grayscale version
   img1 = cv2.imread("SIFT1_img.jpg")
   img2 = cv2.imread("SIFT2_img.jpg")
   gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
   gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

   #run sift algorithm on grayscale image
   sift = cv2.SIFT_create()
   kp1, des1 = sift.detectAndCompute(gray1, None)
   kp2, des2 = sift.detectAndCompute(gray2, None)

   #draw keypoints over color images
   img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
   img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

   #save color images with keypoints
   cv2.imwrite("sift_keypoints_img1.jpg", img1_kp)
   cv2.imwrite("sift_keypoints_img2.jpg", img2_kp)

   #match keypoints between 2 images
   bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
   matches = bf.match(des1, des2)
   matches = sorted(matches, key=lambda x: x.distance)

   #keep top 10% of matches
   top_10_percent = int(len(matches) * 0.10)
   best_matches = matches[:top_10_percent]

   #draw top 10% matches over color images, save to file
   matched_img = cv2.drawMatches(img1, kp1, img2, kp2, best_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   cv2.imwrite("top_10_percent_matches.jpg", matched_img)


if __name__ == "__main__":
   main()