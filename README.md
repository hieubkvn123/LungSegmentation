# LungSegmentation
This repository attempts to perform lung segmentation on chest x-ray images using purely image preprocessing and deep learning approaches like U-Net. The segmentation of lung regions can be later used to perform abnomaly detection on chest x-ray regions

# Lung segmentation with image preprocessing approach.
This repository makes use of the preprocessing pipeline suggested by the "Lung boundary detection for chest X-ray images classification based on GLCM and probabilistic neural networks" paper ([Link](https://www.sciencedirect.com/science/article/pii/S1877050919315145) ).

## Preprocessing pipeline
![Preprocessing pipeline](./media/lungseg_pipeline.png)

 - The testing script can be ran using the test.py file:
```
	python3 test.py --input <path_to_chest_xray_img_file>
```

## Experimental results
#### 1. Segmentation result using BCET preprocessing
![Lung segmentation BCET](./media/lungseg_opencv_bcet.png)

#### 2. Segmentation result using CLAHE on gray scale image
![Lung segmentation CLAHE](./media/lungseg_opencv_clahe.png)

#### 3. Segmentation result using CLAHE on l-channel of LAB image
![Lung segmentation CLAHE-LAB](./media/lungseg_opencv_clahe_lab.png)