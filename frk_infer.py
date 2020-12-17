import os

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import time
import cv2
from tqdm import tqdm

import tensorflow as tf

# Needed to show segmentation colormap labels

from deeplab.utils import get_dataset_colormap
#from deeplab.utils import labels_alstom

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None, 'Where the model is')
flags.DEFINE_string('image_dir', None, 'Where the image is')
flags.DEFINE_string('save_dir', None, 'Dir for saving results')
flags.DEFINE_string('image_name', None, 'Image name')



class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    #INPUT_SIZE = 513
    #INPUT_SIZE = 321
    INPUT_SIZE = 200

    def __init__(self, model_dir):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        model_filename = FLAGS.model_dir
        with tf.gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        print('Image resized')
        start_time = time.time()
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        print('Image processing finished')
        print('Elapsed time : ' + str(time.time() - start_time))
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


model = DeepLabModel(FLAGS.model_dir)
print('Model created successfully')



def vis_segmentation(image, seg_map):
    
    #seg_image = get_dataset_colormap.label_to_color_image(
    #     seg_map, get_dataset_colormap.get_alstom_name()).astype(np.uint8)
    seg_image = get_dataset_colormap.label_to_color_image(seg_map).astype(np.uint8)         
         
    return seg_image



def run_demo_image(image_path):
    try:
        print(image_path)
        orignal_im = Image.open(image_path)

    except IOError:
        print ('Failed to read image from %s.' % image_path)
        return
    print ('running deeplab on image...')
    resized_im, seg_map = model.run(orignal_im)

    return vis_segmentation(resized_im, seg_map)



IMAGE_DIR = FLAGS.image_dir


files = os.listdir(FLAGS.image_dir)
for f in tqdm(files):

    prediction = run_demo_image(IMAGE_DIR+f)
    Image.fromarray(prediction).save(FLAGS.save_dir+'prediction_'+f)
