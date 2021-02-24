import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np

pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

config_file = "/content/drive/MyDrive/exp_hrsc/config.yml"

# update the config options with the config file
cfg.merge_from_file(config_file)

cfg.MODEL.WEIGHT = '/content/drive/MyDrive/exp_hrsc/model_final.pth'  #this is the output .pth file
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.8,
)

val_path='/content/drive/MyDrive/HRSC2016/test_cut/images/' #this is the validation image data
imglistval = os.listdir(val_path) 
i= 0 
for name in imglistval:
    imgfile = val_path + name
    pil_image = Image.open(imgfile).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    

    predictions = coco_demo.run_on_opencv_image(image) # forward predict
    """
    plt.subplot(1, 2, 1)
    plt.imshow(image[:,:,::-1])
    plt.axis('off')
 
    plt.subplot(1, 2, 2)
    plt.imshow(predictions[:,:,::-1])
    plt.axis('off')
    plt.show()
    """
    plt.imsave("/content/drive/MyDrive/exp_hrsc/result/ship"+ str(i) + ".png",predictions[:,:,::-1])
    i=i+1
