import groundingdino.datasets.transforms as T
import numpy as np
import torch
from torchvision.ops import box_convert, nms
from groundingdino.util.inference import load_image, load_model, predict
from segment_anything import build_sam, SamPredictor
from PIL import Image


class Grounded_SAM:

    def __init__(self, classes, box_threshold=0.2, text_threshold=0.2, device="cuda"):
        TEXT_PROMPT = ""
        for obj in classes:
            TEXT_PROMPT += f"{obj.lower()}. "
        TEXT_PROMPT = TEXT_PROMPT.strip()
        self.TEXT_PROMPT = TEXT_PROMPT

        SAM_CHECKPOINT = "/home/zongtai/project/Checkpoints/sam_vit_h_4b8939.pth"

        SAM2_CHECKPOINT = "/home/zongtai/project/Codes/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

        GROUNDING_DINO_CONFIG = "/home/zongtai/project/Codes/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
        GROUNDING_DINO_CHECKPOINT = "/home/zongtai/project/Checkpoints/groundingdino_swinb_cogcoor.pth"

        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.DEVICE = device

        self.sam_model = build_sam(checkpoint=SAM_CHECKPOINT).to(self.DEVICE)
        self.sam_model = self.sam_model.to(self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam_model)

        # self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=self.DEVICE)
        # self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.DEVICE
        )
        self.grounding_model = self.grounding_model.eval()


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

        self.sam_predictor.set_image(img)
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

        if boxes.shape[0] == 0:
            return boxes, confidences, labels, torch.zeros((0, h, w))

        boxes_xyxy = boxes
        nms_idx = nms(
            boxes_xyxy,
            confidences,
            0.8
        ).numpy().tolist()

        boxes_xyxy = boxes_xyxy[nms_idx,:]
        boxes = boxes[nms_idx,:]
        confidences = confidences[nms_idx]
        labels = np.array(labels)[nms_idx].tolist()

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, img.shape[:2]).to(self.DEVICE)
        masks, scores, logits = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels

        return boxes, confidences, class_names, masks