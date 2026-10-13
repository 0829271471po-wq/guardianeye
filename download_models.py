# -*- coding: utf-8 -*-
"""
download_models.py — ดาวน์โหลดไฟล์โมเดล AI ทั้ง 3 ตัว มาไว้ในโฟลเดอร์ models/

รันครั้งเดียวหลังติดตั้งไลบรารี:  python download_models.py
"""
import os

import requests

import config

MODELS = {
    # ตรวจจับใบหน้า (OpenCV Zoo)
    config.FACE_DETECTOR_MODEL:
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    # จดจำใบหน้า (OpenCV Zoo)
    config.FACE_RECOGNIZER_MODEL:
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    # หาโครงร่างกาย (MediaPipe)
    config.POSE_MODEL_PATH:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def download(url, path):
    """ดาวน์โหลดไฟล์จาก url ไปเก็บที่ path"""
    print(f"กำลังดาวน์โหลด {os.path.basename(path)} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()          # ถ้าดาวน์โหลดพลาด ให้แจ้ง error
    with open(path, "wb") as f:
        f.write(response.content)
    print(f"  เสร็จ ({len(response.content) / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    os.makedirs(os.path.join(config.BASE_DIR, "models"), exist_ok=True)
    for path, url in MODELS.items():
        # ไฟล์ที่เล็กผิดปกติ (< 100 KB) อาจดาวน์โหลดมาไม่ครบ → โหลดใหม่
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            print(f"มีอยู่แล้ว: {os.path.basename(path)}")
            continue
        download(url, path)
    print("โมเดลพร้อมใช้งาน ✅")
