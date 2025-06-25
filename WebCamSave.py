# Ultralytics YOLOv5 🚀  + LK Optical‑Flow Tracking
# --------------------------------------------------
"""
此版本在原版 detect.py 基础上加入按下 `t` 后用 Lucas‑Kanade 金字塔光流
追踪当前检测框中心周围的角点，并在画面上叠加轨迹。
其它检测 / 保存逻辑保持不变，方便直接替换运行。
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from utils.plots import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# -----------------------------------------------------------------------------
# LK 参数
LK_PARAMS = dict(winSize=(25, 25),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


@smart_inference_mode()
def run(
    weights=ROOT / "best.pt",
    source=0,
    data=ROOT / "selfdriving.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=True,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    # -----------------------------  追踪状态变量  -----------------------------
    last_det: list | None = None          # 最近一次 NMS 后的结果 (list[Tensor])
    tracking_mode = False                 # 是否进入光流追踪
    old_gray: np.ndarray | None = None    # 前一帧灰度图
    pts_prev: np.ndarray | None = None    # 前一帧有效角点  shape(N,1,2)
    mask: np.ndarray | None = None        # 绘制轨迹的透明层
    colors_pts = None  # <-- 每个角点的 (B,G,R)


    # -----------------------------  数据加载相关  -----------------------------
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------  加载模型  ---------------------------------
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # -----------------------------  Dataloader  -------------------------------
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    dt = (Profile(device=device), Profile(device=device), Profile(device=device))

    # -----------------------------  主循环  ------------------------------------
    for path, im, im0s, vid_cap, s in dataset:
        # ---------- 推理阶段 （如果未在追踪） ----------
        if not tracking_mode:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()
                im /= 255
                if im.ndim == 3:
                    im = im[None]
            input_shape = im.shape[2:]

            with dt[1]:
                pred = model(im, augment=augment, visualize=visualize)
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                last_det = pred
        else:
            pred = last_det

        # ---------- 处理每一帧 ----------
        for i, det in enumerate(pred):
            im0 = im0s.copy() if isinstance(im0s, np.ndarray) else im0s[i].copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(input_shape, det[:, :4], im0.shape).round()
                if not tracking_mode:
                    for *xyxy, conf, cls in det:
                        label = None if hide_labels else (names[int(cls)] if hide_conf else f"{names[int(cls)]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

            im0 = annotator.result()  # 检测框完成，下面可以叠加光流

            # ---------- 如果在追踪则更新并画轨迹 ----------

            if tracking_mode and pts_prev is not None and len(pts_prev) > 0:
                frame_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
                pts_next, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, pts_prev, None, **LK_PARAMS)

                # if pts_next is None:
                #     tracking_mode = False
                #     print("[WARN] LK lost, back to detect")

                if pts_next is None or np.count_nonzero(st) < 5:
                    tracking_mode = False
                    print("[WARN] tracking failed or too few points")
                    continue
                else:
                    good_new = pts_next[st == 1]
                    good_old = pts_prev[st == 1]
                    good_col = colors_pts[st.flatten() == 1]  # 关键行！

                    for (p_new, p_old, col) in zip(good_new, good_old, good_col):
                        a, b = map(int, p_new.ravel())
                        c, d = map(int, p_old.ravel())
                        col = tuple(int(x) for x in col)  # uint8 → int
                        mask = cv2.line(mask, (a, b), (c, d), col, 2)
                        im0 = cv2.circle(im0, (a, b), 4, col[::-1], -1)  # 圆用反序 RGB→BGR

                    im0 = cv2.add(im0, mask)
                    old_gray = frame_gray.copy()
                    pts_prev = good_new.reshape(-1, 1, 2)
                    colors_pts = good_col  # 更新颜色表
                    if len(pts_prev) < 3:
                        tracking_mode = False
                        print("[INFO] few points, stop tracking")

            # ---------- 显示/键盘监听 ----------
            if view_img:
                instruction_color = (0, 255, 255)  # 黄色更显眼 (BGR)
                cv2.putText(im0, "Press  q :  Exit",
                            (10, 20),  # 左上角起点
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, instruction_color, 1, cv2.LINE_AA)

                cv2.putText(im0, "Press  t :  Enter Tracking",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, instruction_color, 1, cv2.LINE_AA)

                cv2.putText(im0, "Press  r :  Back to Detection",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, instruction_color, 1, cv2.LINE_AA)
                cv2.imshow(str(path), im0)


                key = cv2.waitKey(1) & 0xFF

                # # 开启追踪

                if key == ord('t') and not tracking_mode and last_det is not None and len(last_det[0]) > 0:
                    tracking_mode = True
                    old_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros_like(im0)

                    pts_list, col_list = [], []
                    for det_box in last_det[0]:
                        x1, y1, x2, y2, conf, cls = det_box.cpu().numpy()
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        roi = old_gray[max(cy - 20, 0): cy + 20, max(cx - 20, 0): cx + 20]
                        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # roi = old_gray[y1:y2, x1:x2]  # 使用整个bbox而不是中心区域
                        corners = cv2.goodFeaturesToTrack(roi, maxCorners=50, qualityLevel=0.05, minDistance=6)
                        if corners is None:
                            continue
                        # --- 取该类别的专属颜色 (utils.colors 已是 BGR)
                        line_col = colors(int(cls), True)  # tuple( B, G, R )
                        for p in corners:
                            fx, fy = p.ravel()
                            pts_list.append([[fx + cx - 20, fy + cy - 20]])
                            # pts_list.append([[fx + x1, fy + y1]])  # 坐标映射回原图
                            col_list.append(line_col)

                    if len(pts_list) < 5:
                        print("[WARN] too few corners, skip tracking")
                        tracking_mode = False
                        continue

                    pts_prev = np.array(pts_list, dtype=np.float32)
                    colors_pts = np.array(col_list, dtype=np.uint8)  # <-- 同步保存

                # 退出追踪
                elif key == ord('r') and tracking_mode:
                    tracking_mode = False
                    print("[INFO] stop tracking")

                # 退出程序
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5s.pt")
    parser.add_argument("-f","--source", type=str, default=0)
    parser.add_argument("--data", type=str, default=ROOT / "selfdriving.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view-img", action="store_true", default=True)
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("-o","--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())
