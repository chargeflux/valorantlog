# This file is derived from rf-detr
# https://github.com/roboflow/rf-detr
#
# Copyright (c) 2025 Roboflow.
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0

from typing import Iterable, Sequence
import torch
from torchvision.transforms import v2

from valorantlog.inference import Prediction


class RfdetrBase:
    # Copied from https://github.com/roboflow/rf-detr/blob/0c1e8330e41db310e4d86ddf7a5c1b26e7f67489/rfdetr/detr.py#L48
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    def __init__(self, input_h: int, input_w: int):
        self.input_h = input_h
        self.input_w = input_w

    # Adapted from https://github.com/roboflow/rf-detr/blob/0c1e8330e41db310e4d86ddf7a5c1b26e7f67489/rfdetr/detr.py#L222
    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        img = img.float() / 255.0 if (img > 1).any() else img.float()

        transforms = v2.Compose(
            [
                v2.Resize((self.input_w, self.input_h)),
                v2.Normalize(self.means, self.stds),
            ]
        )

        return transforms(img)

    # Copied from https://github.com/roboflow/rf-detr/blob/0c1e8330e41db310e4d86ddf7a5c1b26e7f67489/rfdetr/util/box_ops.py#L22
    def box_cxcywh_to_xyxy(self, x: torch.Tensor) -> torch.Tensor:
        x_c, y_c, w, h = x.unbind(-1)
        b = [
            (x_c - 0.5 * w.clamp(min=0.0)),
            (y_c - 0.5 * h.clamp(min=0.0)),
            (x_c + 0.5 * w.clamp(min=0.0)),
            (y_c + 0.5 * h.clamp(min=0.0)),
        ]
        return torch.stack(b, dim=-1)

    # Adapted from https://github.com/roboflow/rf-detr/blob/0c1e8330e41db310e4d86ddf7a5c1b26e7f67489/rfdetr/models/lwdetr.py#L704
    def postprocess(
        self, img_shape: Sequence[int], boxes: torch.Tensor, logits: torch.Tensor
    ) -> Iterable[Prediction]:
        prob = logits.sigmoid()
        scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), 300, dim=1)
        topk_boxes = topk_indexes // logits.shape[2]
        label_ids = topk_indexes % logits.shape[2]

        boxes = self.box_cxcywh_to_xyxy(boxes)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = torch.tensor([img_shape[1:]]).unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "label_ids": l, "boxes": b}
            for s, l, b in zip(scores, label_ids, boxes)
        ]

        for result in results:
            scores = result["scores"]
            label_ids = result["label_ids"]
            boxes = result["boxes"]
            assert len(scores) == len(label_ids) == len(boxes)

            keep = scores > 0.5
            scores = scores[keep]
            label_ids = label_ids[keep]
            boxes = boxes[keep]
            for i in range(len(boxes)):
                yield Prediction(
                    boxes[i].float().cpu().numpy(),
                    scores[i].float().cpu().numpy().item(),
                    label_ids[i].cpu().numpy().item(),
                )
