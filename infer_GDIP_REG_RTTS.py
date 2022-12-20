from torch.utils.data import DataLoader
import utils.gpu as gpu
from model.yolov3_multilevel_gdip import Yolov3
from tqdm import tqdm
from utils.tools import *
from eval.evaluator_RTTS_GDIP import Evaluator
import argparse
import os
import config.yolov3_config_RTTS as cfg
from utils.visualize import *
from tqdm import tqdm


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'


class Tester(object):
    def __init__(self,
                 weight_path=None,
                 gpu_id=0,
                 img_size=544,
                 visiual=None,
                 eval=False
                 ):
        self.img_size = img_size
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]

        self.__visiual = visiual
        self.__eval = eval
        self.__classes = cfg.DATA["CLASSES"]

        self.__model = Yolov3(cfg).to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, visiual=False)


    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        # self.__model.load_state_dict(chkpt['model'])
        print("loading weight file is done")
        del chkpt


    def test(self):
        if self.__visiual:
            imgs = os.listdir(self.__visiual)
            for v in tqdm(imgs):
                path = os.path.join(self.__visiual, v)
                # print("test images : {}".format(path))

                img = cv2.imread(path)
                assert img is not None

                bboxes_prd = self.__evalter.get_bbox(img)
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]

                    visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
                    path = os.path.join(cfg.PROJECT_PATH, "results/rtts/{}".format(v))
                    cv2.imwrite(path, img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='best.pt', help='weight file path')
    parser.add_argument('--visiual', type=str, default='path/to/images', help='test data path or None')
    parser.add_argument('--eval', action='store_true', default=True, help='eval the mAP or not')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester( weight_path=opt.weight_path,
            gpu_id=opt.gpu_id,
            eval=opt.eval,
            visiual=opt.visiual).test()
