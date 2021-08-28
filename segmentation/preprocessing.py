import cv2
import numpy as np

# Contrast Limited Adaptive Histogram Equalization (CLAHE)
def clahe_lab(img):
    # Median blurring
    img = cv2.medianBlur(img, 3)

    # Convert image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split each channel
    l, a, b = cv2.split(lab)

    # Apply CLAHE to l channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    l = clahe.apply(l)

    # Merge back to LAB
    lab = cv2.merge((l, a, b))

    # Convert back to BGR
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img

# CLAHE
def clahe(img):
    # Median blurring
    img = cv2.medianBlur(img, 7)

    # Apply CLAHE on whole image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    output = clahe.apply(gray)

    img[:,:,0] = output
    img[:,:,1] = output
    img[:,:,2] = output

    return img

# Balanced Contrast Enhancement Technique (BCET)
def bcet(img):
    # Median blurring
    img = cv2.medianBlur(img, 3)

    # Convert image to double
    x    = img.astype('float32')
    Lmin = np.min(x) # Image minimum value
    Lmax = np.max(x) # Image maximum value
    Lmean = np.mean(x) # Image mean value
    LMssum = np.mean(x**2) # Image mean sum of squares

    Gmin = 0 # Min of output image
    Gmax = 255 # Max of output image
    Gmean = 110 # Mean of output image

    bnum = (Lmax**2)*(Gmean-Gmin) - LMssum*(Gmax-Gmin) + (Lmin**2)*(Gmax-Gmean)
    bden = 2*(Lmax*(Gmean-Gmin)-Lmean*(Gmax-Gmin)+Lmin*(Gmax-Gmean))

    b = bnum/bden

    a = (Gmax-Gmin)/((Lmax-Lmin)*(Lmax+Lmin-2*b))

    c = Gmin - a*((Lmin-b)**2)

    y = a * ((x - b)**2) + c # PARABOLIC FUNCTION
    img = y.astype(np.uint8)

    return img
