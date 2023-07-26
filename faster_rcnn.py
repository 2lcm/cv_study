import torch.nn as nn
from resnet50 import ResNet50
from utils import print_tensor

class FasterRCNN(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = self.get_backbone(backbone)
        self.RPN = self.make_RPN()
    
    def forward(self, img, gt_bbox):
        # get feature
        features = self.backbone(img)
        img_size = img.size(2)
        feat_size = features.size(2)

        val = (feat_size / img_size) * gt_bbox
        print(gt_bbox)
        print(val)
        raise NotImplementedError

        # rpn
        anchors, bbox_reg = self.RPN(features) # im_b, h, w, 9 // im_b, h, w, 9 * 4
        rois = self.get_rois(anchors, bbox_reg) # roi mini batches (pos and neg)
        # fast rcnn
        roi_pooling_feature = self.roi_pooling(rois)
        out = self.tail(roi_pooling_feature)
        class_out = self.class_fc(out)
        bbox_reg_out = self.bbox_fc(out)

        return class_out, bbox_reg_out
    
    def get_backbone(self, model_name):
        if model_name == "vgg16":
            from torchvision.models import vgg16
            model = vgg16(pretrained = True)
            return model.features
        else:
            raise NotImplementedError("No such model in code")

    def make_RPN(self):
        return None

if __name__ == "__main__":
    fasterrcnn = FasterRCNN("vgg16", 100)

