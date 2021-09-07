from unet import UnetSegmenter

class Preprocessor:
    def __init__(self, unet_model_file, unet_weights_file):
        self.segmenter = UnetSegmenter(unet_model_file, unet_weights_file)

    def lung_region_normalization(self, img): 

