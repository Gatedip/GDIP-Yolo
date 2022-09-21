# coding=utf-8
DATA_PATH = "/scratch/data"
PROJECT_PATH = "/scratch/"
WEIGHT_PATH="/scratch/data/weights/darknet53_448.weights"

# DATA = {"CLASSES":['person','bicycle','car','bus','motorbike'],
#         "NUM":5}
DATA = {"CLASSES":['bicycle','boat','bottle','bus','car','cat','chair','dog','motorbike','person'],
       "NUM":10}
# model
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }

# train
TRAIN = {
         "TRAIN_IMG_SIZE":448,
         "AUGMENT":True,
         "BATCH_SIZE":8,
         "MULTI_SCALE_TRAIN":False,
         "IOU_THRESHOLD_LOSS":0.5,
         "EPOCHS":80,
         "NUMBER_WORKERS":5,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1e-4,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":2  # or None
         }


# test
TEST = {
        "TEST_IMG_SIZE":448,
        "BATCH_SIZE":1,
        "NUMBER_WORKERS":0,
        "CONF_THRESH":0.01,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False,
        "DATASET_PATH":"/scratch/data/ExDark_VOC/testset",
        "DATASET_DIRECTORY":"JPEGImages"
        }
