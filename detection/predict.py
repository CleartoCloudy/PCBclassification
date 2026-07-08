"""
检测模型推理 + 可视化（在原图上画预测框）
Author: CleartoCloudy
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, CLASS_NAMES, DEVICE, BASE_DIR
from detection.model import create_model

# 缺陷颜色映射
CLASS_COLORS = {
    0: (0, 255, 0),       # open → 绿
    1: (255, 0, 0),       # short → 蓝
    2: (0, 255, 255),     # mousebite → 黄
    3: (255, 0, 255),     # spur → 紫
    4: (255, 255, 0),     # pinhole → 青
    5: (0, 128, 255),     # spurious_copper → 橙
}


def load_detection_model(model_path=None):
    """加载训练的检测模型"""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "detection_faster_rcnn_best.pth")

    model = create_model(num_classes=len(CLASS_NAMES) + 1, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_path, score_thresh=0.3, iou_thresh=0.3):
    """
    对单张 PCB 大图进行缺陷检测
    返回: (image_np, detections)
      detections: [{"box": [x1,y1,x2,y2], "score": float, "class": str, "class_id": int}, ...]
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # 先按分数预过滤，再 NMS（大量低分框没必要参与 NMS）
    score_mask = predictions["scores"] > 0.05
    boxes = predictions["boxes"][score_mask]
    scores = predictions["scores"][score_mask]
    labels = predictions["labels"][score_mask]

    # NMS
    keep = torchvision_nms(boxes, scores, iou_thresh)
    boxes = boxes[keep].cpu()
    scores = scores[keep].cpu()
    labels = labels[keep].cpu()

    # 过滤低置信度
    mask = scores > score_thresh
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        cls_id = int(label) - 1  # 模型输出 1-6，转回 0-5
        if 0 <= cls_id < len(CLASS_NAMES):
            detections.append({
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "score": float(score),
                "class": CLASS_NAMES[cls_id],
                "class_id": cls_id,
            })

    # 转 numpy 绘图
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_id"], (200, 200, 200))

        # 画框
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

        # 标签
        label_text = f"{det['class']} {det['score']:.3f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_np, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image_np, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image_np, detections


def torchvision_nms(boxes, scores, iou_thresh):
    """torchvision NMS 包装"""
    from torchvision.ops import nms
    return nms(boxes, scores, iou_thresh)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="图像路径或目录")
    parser.add_argument("--threshold", type=float, default=0.3, help="置信度阈值")
    args = parser.parse_args()

    print(f"加载检测模型...")
    model = load_detection_model()
    print(f"设备: {DEVICE}")

    out_dir = os.path.join(BASE_DIR, "results", "reports",
                           f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(args.image):
        image_paths = sorted([
            os.path.join(args.image, f) for f in os.listdir(args.image)
            if f.endswith((".jpg", ".png", ".bmp"))
        ])
    else:
        image_paths = [args.image]

    for img_path in image_paths:
        print(f"\n检测: {os.path.basename(img_path)}")
        try:
            img_result, detections = predict_image(model, img_path, score_thresh=args.threshold)

            print(f"  发现 {len(detections)} 个缺陷:")
            for det in sorted(detections, key=lambda d: d["score"], reverse=True):
                x1, y1, x2, y2 = det["box"]
                print(f"    [{det['class']:20s}] conf={det['score']:.3f}  "
                      f"box=({x1},{y1},{x2},{y2})  size={x2-x1}x{y2-y1}")

            out_path = os.path.join(out_dir, f"det_{os.path.basename(img_path)}")
            cv2.imwrite(out_path, img_result)
            print(f"  已保存: {out_path}")
        except Exception as e:
            print(f"  错误: {e}")


if __name__ == "__main__":
    main()
