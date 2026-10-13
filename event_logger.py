# -*- coding: utf-8 -*-
"""
event_logger.py — บันทึกเหตุการณ์ลงไฟล์ CSV และเก็บรูปตอนเกิดเหตุ

logs/events.csv มีคอลัมน์: timestamp, event_type, person, snapshot_path
รูปเก็บใน snapshots/ ตั้งชื่อตามเวลา เช่น snapshots/20261008_143205_fall.jpg
"""
import csv
import os
from datetime import datetime

import cv2

import config

CSV_HEADER = ["timestamp", "event_type", "person", "snapshot_path"]


def save_snapshot(frame, event_type):
    """บันทึกรูปลงโฟลเดอร์ snapshots/ คืนค่าที่อยู่ไฟล์"""
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{event_type}.jpg"
    path = os.path.join(config.SNAPSHOT_DIR, filename)
    # ใช้ imencode + tofile แทน cv2.imwrite เพราะ imwrite บน Windows
    # บันทึกไม่ได้ถ้าที่อยู่ไฟล์มีตัวอักษรไทย
    ok, buffer = cv2.imencode(".jpg", frame)
    if ok:
        buffer.tofile(path)
    return path


def log_event(event_type, person, frame=None):
    """
    บันทึกเหตุการณ์ 1 รายการ
    event_type เช่น "fall", "stranger"
    person     ชื่อคน เช่น "คุณยาย"
    frame      ภาพตอนเกิดเหตุ (ถ้ามี จะเก็บรูปด้วย)
    """
    snapshot_path = save_snapshot(frame, event_type) if frame is not None else ""

    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    is_new_file = not os.path.exists(config.LOG_FILE)
    # ไฟล์ใหม่ใช้ "utf-8-sig" (ใส่เครื่องหมาย BOM ต้นไฟล์) ทำให้เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน
    encoding = "utf-8-sig" if is_new_file else "utf-8"
    with open(config.LOG_FILE, "a", newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(CSV_HEADER)   # ไฟล์ใหม่ → เขียนหัวตารางก่อน
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type,
            person,
            os.path.relpath(snapshot_path, config.BASE_DIR) if snapshot_path else "",
        ])
    print(f"[logger] บันทึกเหตุการณ์: {event_type} / {person}")
    return snapshot_path
