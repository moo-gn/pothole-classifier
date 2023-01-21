from PIL import Image
from depth_estimator import DepthEstimator
from utils import visualize_depth_map, point_cloud
import numpy as np
from matplotlib import pyplot as plt

model = DepthEstimator()
imgpath = ''
with Image.open(imgpath) as im:
        img = im
        img = img.convert('RGB')
        # print(type(im))
        img = img.resize((256, 256))
predimg = DepthEstimator.classify(model, img)
# print(predimg.shape)
visualize_depth_map(np.array(img), predimg)
# point_cloud(predimg,np.array(img))

# data = np.load(imgpath)
# print(data.shape)
# plt.imshow(predimg.squeeze(), interpolation='nearest')
# plt.show()