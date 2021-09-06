import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

class UnetSegmenter:
    def __init__(self, model_file, weights_file):
        if(not os.path.exists(model_file) or not os.path.exists(weights_file)):
            raise Exception("Model file or weights file does not exist")

        self.model = load_model(model_file)
        self.model.load_weights(weights_file)

    def img_preprocess(self, img):
        mean = np.mean(img)
        std = np.std(img)

        img_norm = (img - mean) / std

        return img_norm

    def visualize_prediction(self, img, out_file='output.png'):
        # 1.1. Preprocess image
        img = self.img_preprocess(img)

        # 1.2. Generate prediction
        pred = self.model.predict(np.array([img]))[0]
        pred[pred >= 0.75] = 1.0
        pred[pred < 0.75] = 0.0
        pred = (pred * 255.0).astype(np.uint8)

        # 2.1. Find contours
        contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 2.2. Convex hull
        hull_list = []
        hull_areas = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
            hull_area = cv2.contourArea(hull)
            hull_areas.append(hull_area)

        hull_areas = np.array(hull_areas)
        hull_areas = np.sort(hull_areas)
        top_areas = np.array(hull_areas) # [-5:-1]
        # 3. Draw contours and hull result
        drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (np.random.randint(127, 256), np.random.randint(127, 256),
                     np.random.randint(127,256))

            # Filter contour by area
            cnt_area = cv2.contourArea(hull_list[i])
            if(cnt_area not in top_areas):
                continue

            cv2.drawContours(drawing, contours, i, color)
            cv2.drawContours(drawing, hull_list, i, color, thickness=cv2.FILLED)

        # 4. Visualize the prediction result
        fig, ax = plt.subplots(1, 3, figsize=(21, 8))
        ax[0].imshow(img)
        ax[0].set_title('Original Image')

        ax[1].imshow(pred)
        ax[1].set_title('Unet predicted segmentation mask')

        ax[2].imshow(drawing)
        ax[2].set_title('Segmentation mask')

        plt.savefig(out_file)

