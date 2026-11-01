"""Find SAM2 box prompts for the actor (object id 0) and the manipulated object (id 1) with Grounding DINO.

Candidate frames are scanned in order and the first one where both are detected is used. The actor is the
largest person box; the object is the confident box closest to the actor. Grounding DINO runs on the CPU,
which is fast enough for a handful of frames.

Usage: python -m hoi_imitation.reconstruction.prompts --video_path V --object_prompt "chair" \
           --model_dir WEIGHTS --output_path prompts.json
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import groundingdino
import torch
from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
from v2d.grounding_dino.lib.image_list_to_object_bboxes import _preprocess_image

Box = list[float]  # x0, y0, x1, y1 in pixels
Detection = tuple[Box, float]


def detect(model, rgb, prompt: str, box_threshold: float) -> list[Detection]:
    """All boxes above ``box_threshold`` with their confidence."""
    height, width = rgb.shape[:2]
    boxes, logits, _ = predict(
        model=model,
        image=_preprocess_image(rgb),
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=0.25,
        device="cpu",
    )
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy") * torch.tensor([width, height, width, height])
    return [([float(v) for v in box], float(score)) for box, score in zip(xyxy, logits, strict=True)]


def box_area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def pick_actor(people: list[Detection]) -> Detection | None:
    """The actor is the largest person in view; bystanders are smaller and farther away."""
    return max(people, key=lambda detection: box_area(detection[0])) if people else None


def pick_object(objects: list[Detection], actor: Detection | None, image_width: int) -> Detection | None:
    """Among confident boxes, the one closest to the actor; boxes spanning most of the frame are rejected."""
    objects = [d for d in objects if d[0][2] - d[0][0] < 0.6 * image_width]
    if not objects or actor is None:
        return None
    best = max(score for _, score in objects)
    actor_x, actor_y = box_center(actor[0])

    def distance(detection: Detection) -> float:
        x, y = box_center(detection[0])
        return (x - actor_x) ** 2 + (y - actor_y) ** 2

    return min((d for d in objects if d[1] >= 0.6 * best), key=distance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--object_prompt", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 15, 30, 60, 90, 120, 180, 240, 300])
    parser.add_argument("--box_threshold", type=float, default=0.3)
    args = parser.parse_args()

    config = os.path.join(groundingdino.__path__[0], "config", "GroundingDINO_SwinT_OGC.py")
    model = load_model(config, os.path.join(args.model_dir, "groundingdino_swint_ogc.pth"), device="cpu")
    capture = cv2.VideoCapture(args.video_path)
    report = []
    for frame_index in args.frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        actor = pick_actor(detect(model, rgb, "person .", args.box_threshold))
        target = pick_object(detect(model, rgb, args.object_prompt + " .", args.box_threshold), actor, rgb.shape[1])
        report.append({"frame": frame_index, "actor": actor, "object": target})
        print(report[-1], flush=True)
        if actor and target:
            prompts = {
                "prompts": [
                    {
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "box": dict(zip(("x0", "y0", "x1", "y1"), box, strict=True)),
                    }
                    for object_id, (box, _) in ((0, actor), (1, target))
                ]
            }
            with open(args.output_path, "w") as handle:
                json.dump(prompts, handle, indent=2)
            for (box, _), color in ((actor, (0, 255, 0)), (target, (0, 0, 255))):
                cv2.rectangle(bgr, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), color, 3)
            cv2.imwrite(os.path.splitext(args.output_path)[0] + ".jpg", bgr)
            return
    raise SystemExit(f"no frame with both detections: {report}")


if __name__ == "__main__":
    main()
