import groundingdino.datasets.transforms as T
import imageio.v2 as imageio
import numpy as np
import os
import supervision as sv
import torch
from ultralytics import YOLO
from groundingdino.util.inference import load_image, load_model, predict
from new_object_list import categories
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from tqdm import tqdm
from glee_detector import *
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import inference_ram_openset as inference_openset
from ram import get_transform


class Grounded_SAM:

    def __init__(self, classes, box_threshold=0.2, text_threshold=0.2, device="cuda"):
        TEXT_PROMPT = ""
        for obj in classes:
            TEXT_PROMPT += f"{obj.lower()}. "
        TEXT_PROMPT = TEXT_PROMPT.strip()
        self.TEXT_PROMPT = TEXT_PROMPT

        SAM2_CHECKPOINT = "/home/zongtai/project/Codes/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        GROUNDING_DINO_CONFIG = "/home/zongtai/project/Codes/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = "/home/zongtai/project/Codes/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"

        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.DEVICE = device

        self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=self.DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.DEVICE
        )


    def segment(self, img):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        pil_img = Image.fromarray(img)
        img_transformed, _ = transform(pil_img, None)

        self.sam2_predictor.set_image(img)
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=img_transformed,
            caption=self.TEXT_PROMPT,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            remove_combined=True,
            device=self.DEVICE
        )

        h, w, _ = img.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels

        return boxes, confidences, class_names, masks


class GLEE:
    def __init__(self, classes, box_threshold=0.2, text_threshold=0.2, device="cuda"):
        self.classes = classes

        GLEE_CONFIG_PATH = "/home/zongtai/project/Codes/VLN/thirdparty/GLEE/configs/SwinL.yaml"
        GLEE_CHECKPOINT_PATH = "/home/zongtai/project/Codes/VLN/thirdparty/GLEE/GLEE_SwinL_Scaleup10m.pth"

        self.device = device
        self.glee_model = initialize_glee(GLEE_CONFIG_PATH, GLEE_CHECKPOINT_PATH, device)

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def segment(self, img):
        pred_bboxes, pred_masks, pred_class, pred_confidence = glee_segmentation(
            img,
            self.glee_model,
            custom_category=self.classes,
            threshold_select=self.box_threshold,
            device=self.device
        )

        return torch.Tensor(pred_bboxes), pred_confidence, pred_class, pred_masks


class YOLOv11:
    def __init__(self, classes, box_threshold=0.2, text_threshold=0.2, device="cuda"):
        self.classes = classes

        self.device = device
        self.model = YOLO("yolo11x-seg.pt")

    def segment(self, img):
        results = self.model("/home/zongtai/project/Data/HM3d/sample_imgs/473-rgb.jpg")

        print(results)
        exit(0)

        boxes = results.xyxy[0]
        confidences = results.xyxy[0][:, 4]
        class_ids = results.xyxy[0][:, 5]
        masks = results.pred[0]

        # class_names = []
        # for class_id in class


def visualize_mask(img, boxes, confidences, class_names, masks):

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    # input_boxes = boxes.numpy()

    class_ids = np.array(list(range(len(class_names))))
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    return annotated_frame


if __name__ == "__main__":
    DATA_PATH = "/home/zongtai/project/Data/HM3d/sample_imgs"

    imgs = []
    for i in range(500):
        nwimg = imageio.imread(f"{DATA_PATH}/{i}-rgb.jpg")
        imgs.append(nwimg)

    depths = []
    for i in range(500):
        nwdepth = imageio.imread(f"{DATA_PATH}/{i}-depth.jpg")
        nwdepth = nwdepth[:, :, 0]
        depths.append(nwdepth)

    positions = []
    with open(f"{DATA_PATH}/position.txt", "r") as f:
        for line in f:
            line = line[1:-2]
            pos = [float(x) for x in line.strip().split()]
            positions.append(pos)

    rotations = []
    with open(f"{DATA_PATH}/rotation.txt", "r") as f:
        for line in f:
            line = line[11:-2]
            rot = [float(x) for x in line.strip().split(',')]
            rotations.append(rot)

    classes = []
    for obj in categories:
        classes.append(obj['name'])
    print("number of classes: ", len(classes))


    # # SAM
    # model_sam = Grounded_SAM(classes)

    # # write video
    # os.makedirs("outputs/sam2", exist_ok=True)
    # sam_writer = imageio.get_writer("outputs/sam2.mp4", fps=1)

    # for i in tqdm(range(len(imgs))):
    #     img = imgs[i]
    #     boxes, confidences, class_names, masks = model_sam.segment(img)
    #     res = visualize_mask(img, boxes, confidences, class_names, masks)

    #     res_img = np.zeros((res.shape[0], res.shape[1] * 2, 3), dtype=np.uint8)
    #     res_img[:, :res.shape[1]] = img
    #     res_img[:, res.shape[1]:] = res

    #     sam_writer.append_data(res_img)

    # sam_writer.close()

    # # GLEE
    # model_glee = GLEE(classes)

    # # write video
    # os.makedirs("outputs/glee", exist_ok=True)
    # glee_writer = imageio.get_writer("outputs/glee.mp4", fps=1)

    # for i in tqdm(range(len(imgs))):
    #     img = imgs[i]
    #     boxes, confidences, class_names, masks = model_glee.segment(img)
    #     res = visualize_mask(img, boxes, confidences, class_names, masks)

    #     res_img = np.zeros((res.shape[0], res.shape[1] * 2, 3), dtype=np.uint8)
    #     res_img[:, :res.shape[1]] = img
    #     res_img[:, res.shape[1]:] = res

    #     glee_writer.append_data(res_img)

    # glee_writer.close()

    # # YOLO
    # model_yolo = YOLOv11(classes)

    # # write video
    # os.makedirs("outputs/yolo", exist_ok=True)
    # yolo_writer = imageio.get_writer("outputs/yolo.mp4", fps=1)

    # for i in tqdm(range(len(imgs))):
    #     img = imgs[i]
    #     boxes, confidences, class_names, masks = model_yolo.segment(img)
    #     res = visualize_mask(img, boxes, confidences, class_names, masks)

    #     res_img = np.zeros((res.shape[0], res.shape[1] * 2, 3), dtype=np.uint8)
    #     res_img[:, :res.shape[1]] = img
    #     res_img[:, res.shape[1]:] = res

    #     yolo_writer.append_data(res_img)

    # yolo_writer.close()

    # RAM + SAM

    model_ram = ram_plus(
        pretrained="/home/zongtai/project/Checkpoints/ram_plus_swin_large_14m.pth",
        image_size=384,
        vit="swin_l")
    model_ram.eval()
    model_ram = model_ram.to("cuda")

    model_sam = Grounded_SAM(classes)

    transform = get_transform(image_size=384)

    # write video
    os.makedirs("outputs/ram_plus_sam2", exist_ok=True)
    sam_writer = imageio.get_writer("outputs/ram_plus_sam2.mp4", fps=1)

    for i in tqdm(range(len(imgs))):
    # for i in range(341, 342):
        img = imgs[i]

        nwimg = np.zeros((img.shape[1], img.shape[1], 3), dtype=np.uint8)
        nwimg[80:560, :, :] = img
        nwimg = transform(Image.fromarray(nwimg)).unsqueeze(0).to("cuda")

        res = inference(nwimg, model_ram)
        tag_list = res[0].split(" | ")
        tag_list.append("nothing")
        print(tag_list)
        model_sam.TEXT_PROMPT = ". ".join(tag_list) + "."

        boxes, confidences, class_names, masks = model_sam.segment(img)
        res = visualize_mask(img, boxes, confidences, class_names, masks)

        res_img = np.zeros((res.shape[0], res.shape[1] * 2, 3), dtype=np.uint8)
        res_img[:, :res.shape[1]] = img
        res_img[:, res.shape[1]:] = res

        sam_writer.append_data(res_img)

    sam_writer.close()