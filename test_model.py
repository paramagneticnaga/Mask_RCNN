from os import listdir
from xml.etree import ElementTree
import numpy as np
from mrcnn.config import Config
import matplotlib.pyplot as plt
from numpy import expand_dims
from numpy import mean
from matplotlib.patches import Rectangle
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as pyplot
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


class MaskDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "with_mask")
        self.add_class("dataset", 2, "without_mask")
        # define data locations
        images_dir = dataset_dir + "/images /"
        annotations_dir = dataset_dir + "/annots/"
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 800:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 800:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + ".xml"
            # add to dataset
            self.add_image(
                "dataset",
                image_id=image_id,
                path=img_path,
                annotation=ann_path,
            )

    # extract bounding boxes from an annotation file
    @staticmethod
    def extract_boxes(filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall(".//bndbox"):
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find(".//size/width").text)
        height = int(root.find(".//size/height").text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info["annotation"]
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype="uint8")
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index("with_mask"))
            class_ids.append(self.class_names.index("without_mask"))
        return masks, np.asarray(class_ids, dtype="int32")

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]


# train set
train_set = MaskDataset()
train_set.load_dataset(
    "/Users/ishansharma/Downloads/Mask_RCNN/mrcnn/original_data", is_train=True,
)
train_set.prepare()
print("Train: %d" % len(train_set.image_ids))

# test/val set
test_set = MaskDataset()
test_set.load_dataset(
    "/Users/ishansharma/Downloads/Mask_RCNN/mrcnn/original_data", is_train=False,
)
test_set.prepare()
print("Test: %d" % len(test_set.image_ids))

_image_id = 0
image = train_set.load_image(_image_id)
print(image.shape)
mask, class_ids = train_set.load_mask(_image_id)
print(mask.shape)

plt.imshow(image)
plt.imshow(mask[:, :, 0], cmap="gray", alpha=0.5)
plt.show()


class MaskConfig(Config):
    NAME = "Mask_cfg"
    NUM_CLASSES = 2 + 1
    STEPS_PER_EPOCH = 131


config = MaskConfig()

model = MaskRCNN(mode="training", model_dir="./", config=config, )
model.load_weights(
    "/Users/ishansharma/Downloads/mask_rcnn_coco.h5",
    by_name=True,
    exclude=[
        "mrcnn_class_logit",
        "mrcnn_bbox_fc",
        "mrcnn_bbox",
        "mrcnn_mask",
        "mrcnn_class_logits",
    ],
)
model.train(
    train_set,
    test_set,
    learning_rate=config.LEARNING_RATE,
    epochs=5,
    layers="heads",
)


class PredictionConfig(Config):
    Name = "Mask_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    @staticmethod
    def evaluate_model(dataset, model, cfg):
        APs = list()
        for image_id in dataset.image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id,
                                                                             use_mini_mask=False)
            scaled_image = mold_image(image, cfg)
            # convert image into one sample
            sample = expand_dims(scaled_image, 0)
            # make prediction
            yhat = model.detect(sample, verbose=0)
            # extract results for first sample
            r = yhat[0]
            # calculate statistics, including AP
            AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            # store
            APs.append(AP)
            # calculate the mean AP across all images
        mAP = mean(APs)
        return mAP

    def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
        # load image and mask
        for i in range(n_images):
            # load the image and mask
            image = dataset.load_image(i)
            mask, _ = dataset.load_mask(i)
            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, cfg)
            # convert image into one sample
            sample = expand_dims(scaled_image, 0)
            # make prediction
            yhat = model.detect(sample, verbose=0)[0]
            # define subplot
            pyplot.subplot(n_images, 2, i * 2 + 1)
            # plot raw pixel data
            pyplot.imshow(image)
            pyplot.title('Actual')
            # plot masks
            for j in range(mask.shape[2]):
                pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
            # get the context for drawing boxes
            pyplot.subplot(n_images, 2, i * 2 + 2)
            # plot raw pixel data
            pyplot.imshow(image)
            pyplot.title('Predicted')
            ax = pyplot.gca()
            # plot each box
            for box in yhat['rois']:
                # get coordinates
                y1, x1, y2, x2 = box
                # calculate width and height of the box
                width, height = x2 - x1, y2 - y1
                # create the shape
                rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                # draw the box
                ax.add_patch(rect)
        # show the figure
        pyplot.show()


train_set = MaskDataset()
train_set.load_dataset('MaskDataset', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
cfg = PredictionConfig()
model1 = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model1.load_weights('/Users/ishansharma/Downloads/mask_rcnn_coco.h5', by_name=True)
train_mAP = PredictionConfig.evaluate_model(train_set, model1, cfg)
print("Train mAP: %3f" % train_mAP)
test_mAP = PredictionConfig.evaluate_model(test_set, model1, cfg)
print("Test mAP: %3f" % test_mAP)



