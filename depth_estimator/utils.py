import numpy as np
from tensorflow.keras.layers import Layer, InputSpec
import matplotlib.pyplot as plt

def depth_norm(x, maxDepth):
    return maxDepth / x


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(depth_norm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(file.reshape(256, 256, 3) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)


def visualize_depth_map(inputs, pred):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")
    fig, ax = plt.subplots(1, 2, figsize=(50, 50))
    ax[0].imshow((inputs.squeeze()))
    ax[1].imshow((pred.squeeze()), cmap=cmap)
    plt.show()