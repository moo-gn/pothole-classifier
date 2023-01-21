import cv2
import numpy as np

def create_edge_mask(img: cv2.Mat):
    """
    Takes an image and returns the contour mask of edge and the arclength of the edge.
    @params:
    img: cv2.Mat
    @output:
    mask: np.ndarray
    arclength: float
    """
    
    # Turn the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # Get Hue and blur it
    channels = cv2.split(hsv)
    H = channels[0]
    H_blur = cv2.medianBlur(H, 31) 
    
    # Turn the grayscale image to black and white
    _, im_bw = cv2.threshold(H_blur, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Blur the image to smooth it
    median_blur = cv2.medianBlur(im_bw, 21)

    # Canny Edge Detection
    edges = cv2.Canny(image=median_blur, threshold1=255/3, threshold2=255)

    # Get the countour shapes in the image
    contours, _ = cv2.findContours(edges, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create mask for biggest contour 
    contour = max(contours, key = len)
    arclength = cv2.arcLength(contour, True)
    mask = np.zeros(img.shape)
    cv2.drawContours(mask, contour, -1, (0, 255, 0), 1)

    # Return mask
    return mask, arclength
