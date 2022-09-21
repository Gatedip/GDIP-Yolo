import sys
sys.path.append("..")
import xml.etree.ElementTree as ET
import config.yolov3_config_voc_ia_yolo as cfg
import os
from tqdm import tqdm


#list_of_cls = ['']
def parse_voc_annotation(data_path, file_type, anno_path, use_difficult_bbox=False):
    """
    pascal voc annotation,[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: D:\doc\data\VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param file_type: 'trainval''train''val'
    :param anno_path: 
    :param use_difficult_bbox:difficult==1 bbox
    """
    classes = cfg.DATA["CLASSES"]
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', file_type+'.txt')
    with open(img_inds_file, 'r') as f:
        lines = f.readlines()
        image_id = [line.strip().split(' ') for line in lines]
    image_ids = list()
    for i in range(len(image_id)):
        # image_id[i].remove('')
        if image_id[i][-1] == '1':
            image_ids.append(image_id[i][0])

    with open(anno_path, 'a') as f:
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, 'JPEGImages', image_id + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_id + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1): 
                    continue
                bbox = obj.find('bndbox')
                if obj.find("name").text.lower().strip() in cfg.DATA["CLASSES"]:
                    class_id = classes.index(obj.find("name").text.lower().strip())
                    xmin = bbox.find('xmin').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
            annotation += '\n'
            # print(annotation)
            f.write(annotation)
    return len(image_ids)


if __name__ =="__main__":
    # train_set :  VOC2007_trainval VOC2012_trainval
    #train_data_path_2007 = os.path.join(cfg.DATA_PATH, 'VOCtrainval-2007', 'VOCdevkit', 'VOC2007')
    #train_data_path_2012 = os.path.join(cfg.DATA_PATH, 'VOCtrainval-2012', 'VOCdevkit', 'VOC2012')
    #train_annotation_path = os.path.join('/scratch/data', 'train_annotation.txt')
    #if os.path.exists(train_annotation_path):
        #os.remove(train_annotation_path)

    # val_set   : VOC2007_test
    test_data_path_2007 = os.path.join(cfg.DATA_PATH, 'ExDark_VOC','testset')
    test_annotation_path = os.path.join('/scratch/data', 'test_annotation.txt')
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)
    #len_train = 0
    len_test = 0
    for text_file in cfg.DATA['CLASSES']:
        #len_train += parse_voc_annotation(train_data_path_2007, text_file+"_trainval", train_annotation_path, use_difficult_bbox=False) + parse_voc_annotation(train_data_path_2012, text_file+"_trainval", train_annotation_path, use_difficult_bbox=False)
        len_test += parse_voc_annotation(test_data_path_2007, text_file+"_test", test_annotation_path, use_difficult_bbox=False)

    print("The number of images for test : {0}".format(len_test))
