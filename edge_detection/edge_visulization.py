import cv2
import numpy as np
 
FILENAME = 'pot2'
EXTENSTION = '.png'

# Read the original image
img = cv2.imread(FILENAME+EXTENSTION) 
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
 
# Turn the image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

channels = cv2.split(hsv)

H = channels[0]
cv2.imshow('H S V', np.concatenate((channels[0], channels[1], channels[2]), axis=1))
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_H_S_V_'+EXTENSTION, np.concatenate((channels[0], channels[1], channels[2]), axis=1))
cv2.imshow('H', H)
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_HUE_'+EXTENSTION, H)
H_blur = cv2.medianBlur(H, 31) 
cv2.imshow('H blurred', H_blur)
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_MEDIAN_BLUR_1_'+EXTENSTION, H_blur)
 
# Turn the grayscale image to black and white
thresh, im_bw = cv2.threshold(H_blur, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('Turned image from grayscale to black/white', im_bw)
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_BINARIZATION_'+EXTENSTION, im_bw)
# Blur the image to smooth it
median_blur = cv2.medianBlur(im_bw, 21)
cv2.imshow('Smoothed image', median_blur )
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_MEDIAN_BLUR_2_'+EXTENSTION, median_blur)
# Canny Edge Detection
edges = cv2.Canny(image=median_blur, threshold1=255/3, threshold2=255) # Canny 
# Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_CANNY_EDGE_'+EXTENSTION, edges)

# Get the countour shapes in the image
contours, hierarchy = cv2.findContours(edges, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw contours in the edges
mask_contours = np.zeros(img.shape)
cv2.drawContours(mask_contours, contours, -1, (0, 255, 0), 1)
cv2.imshow('Contour Detection', mask_contours)
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_CONTOURS_'+EXTENSTION, mask_contours)

# Create mask for biggest contour 
contour = max(contours, key = len)
mask = np.zeros(img.shape)
cv2.drawContours(mask, contour, -1, (0, 255, 0), 1)
cv2.imshow('Biggest contour mask', mask)
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_BIGGEST_CONTOUR_'+EXTENSTION, mask)
  
# Draw the contour with the maximum size
cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
cv2.imshow('Place Contour on original img', img)
cv2.waitKey(0)
cv2.imwrite(FILENAME+'_CONTOUR_ON_IMG_'+EXTENSTION, img)
 
cv2.destroyAllWindows()
