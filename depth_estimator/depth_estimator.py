from PIL import Image
import tensorflow as tf
import numpy as np
from utils import BilinearUpSampling2D, load_images, predict
class DepthEstimator(object):
    def __init__(self) -> None:
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
        print('Loading model...')
        self.model = tf.keras.models.load_model("indoormodel/", custom_objects=custom_objects, compile=False)
        print('Successfully loaded model...')
        pass

    def classify(self, image: Image) -> str:
        image = np.asarray(image)
        inputs = load_images([image])
        outputs = predict(self.model, inputs)
        return outputs

        """Classify severity of a pothole image.
        Args:
            image (Image): Input PIL image of a cropped plothole.
        Returns:
            str: severity score.
        """
        # TODO
        pass