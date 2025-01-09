import supervision as sv
from PIL import Image
from ram import get_transform
from ram import inference_ram as inference
from ram import inference_ram_openset as inference_openset
from ram.models import ram_plus
from supervision.geometry.core import Position
from torchvision.ops import box_convert

from constants import *

from .glee_detector import *
from .sam import Grounded_SAM


def visualize_mask(img, boxes, confidences, class_names, masks):

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    # input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    input_boxes = boxes.numpy()

    class_ids = np.array(list(range(len(class_names))))
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(text_position=Position.CENTER)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    return annotated_frame


class ImagePerceiver:

    def __init__(self, classes, box_threshold=0.2, text_threshold=0.2, device="cuda"):
        self.classes = classes
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def segment(self, img):
        pass

    def perceive(self, img, area_threshold=2500):
        boxes, confidences, classes, masks = self.segment(img)

        try:
            mask_area = np.array([mask.sum() for mask in masks])
            bbox_trust = np.array([(bbox[0] > 20) & (bbox[2] < img.shape[1] - 20) for bbox in boxes
                                  ])
            flag = (mask_area > area_threshold) & bbox_trust
            visualization = visualize_mask(img, boxes[flag], confidences[flag],
                                           classes[flag], masks[flag])
            return classes[flag], masks[flag], confidences[flag], [visualization]
        except:
            return [], [], [], [img]


class GLEE_Perceiver(ImagePerceiver):

    def __init__(self, classes, box_threshold=0.2, text_threshold=0.2, device="cuda"):
        super().__init__(classes, box_threshold, text_threshold, device)
        self.glee_model = initialize_glee(GLEE_CONFIG_PATH, GLEE_CHECKPOINT_PATH, device)

    def segment(self, img):
        pred_bboxes, pred_masks, pred_class, pred_confidence = glee_segmentation(
            img,
            self.glee_model,
            custom_category=self.classes,
            threshold_select=self.box_threshold,
            device=self.device,
        )

        return torch.Tensor(pred_bboxes), pred_confidence, pred_class, pred_masks


class RAMDINOSAM_Perceiver(ImagePerceiver):

    def __init__(self, classes, box_threshold=0.2, text_threshold=0.2, device="cuda"):
        super().__init__(classes, box_threshold, text_threshold, device)

        self.ram = ram_plus(
            pretrained="/home/zongtai/project/Checkpoints/ram_plus_swin_large_14m.pth",
            image_size=384,
            vit="swin_l")
        self.ram.eval()
        self.ram = self.ram.to(device)

        self.sam = Grounded_SAM(classes)
        self.transform = get_transform(image_size=384)

    def segment(self, img):
        nwimg = np.zeros((img.shape[1], img.shape[1], 3), dtype=np.uint8)
        nwimg[80:560, :, :] = img
        nwimg = self.transform(Image.fromarray(nwimg)).unsqueeze(0).to(self.device)

        res = inference(nwimg, self.ram)
        tag_list = res[0].split(" | ")
        tag_list.append("nothing")
        self.sam.TEXT_PROMPT = ". ".join(tag_list) + "."

        return self.sam.segment(img)


class DINOSAM_Perceiver(ImagePerceiver):

    def __init__(self, classes, box_threshold=0.2, text_threshold=0.2, device="cuda"):
        super().__init__(classes, box_threshold, text_threshold, device)

        self.sam = Grounded_SAM(classes)

    def segment(self, img):
        return self.sam.segment(img)


class MMDINOSAM_Perceiver:

    def __init__(self, device):
        pass
