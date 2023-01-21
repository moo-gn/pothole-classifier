from PIL import Image
import tensorflow as tf
import numpy as np
from .utils import BilinearUpSampling2D, load_images, predict
from edge_detection.create_edge_mask import create_edge_mask

class DepthEstimator(object):
    def __init__(self) -> None:
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
        print('Loading model...')
        self.model = tf.keras.models.load_model("depth_estimator/10efinetunedmodel/", custom_objects=custom_objects, compile=False)
        print('Successfully loaded model...')

    def classify(self, image: Image) -> str:
        """Classify severity of a pothole image.
        Args:
            image (Image): Input PIL image of a cropped plothole.
        Returns:
            str: severity score.
        """
        image = image.resize((256, 256))
        input = np.asarray(image, dtype="float32")
        depth_map = predict(self.model, input)
        input = np.array(image)
        mask, arclength = create_edge_mask(input)
        return self.compute_severity(depth_map, mask, arclength), arclength

    def compute_severity(self, depth_array: np.ndarray, mask, arclength) -> float:
        # Multiply the mask with the depth_array
        multip = depth_array * mask

        return np.average(multip)
