# -*- coding: utf-8 -*-
"""
camera.py — อ่านภาพจากกล้อง (webcam / มือถือ / ESP32-CAM)

แนวคิดสำคัญ:
- อ่านภาพใน "thread แยก" (เหมือนมีผู้ช่วยอีกคนคอยรับภาพตลอดเวลา)
  แล้วเก็บไว้แค่ "เฟรมล่าสุด" เฟรมเดียว
- ถ้าอ่านใน loop หลักตรง ๆ แล้วการประมวลผล AI ช้ากว่ากล้อง ภาพจะค้างสะสมในคิว
  ทำให้ภาพที่เห็นช้ากว่าความจริงไปเรื่อย ๆ (ภาพหน่วง) — วิธีนี้แก้ปัญหานั้น
- ถ้ากล้องหลุด จะพยายามเชื่อมต่อใหม่ทุก RECONNECT_DELAY วินาที

ทดสอบไฟล์นี้อย่างเดียว:  python camera.py
"""
import threading
import time
from urllib.parse import urlsplit

import cv2
import requests

import config


class Camera:
    """ตัวอ่านภาพจากกล้องที่ทำงานเบื้องหลัง"""

    def __init__(self, source=config.CAMERA_SOURCE):
        self.source = source
        self.frame = None          # เฟรมล่าสุดที่อ่านได้
        self.frame_id = 0          # เลขนับเฟรม (เพิ่มทุกครั้งที่มีเฟรมใหม่) ใช้เช็กว่าเป็นเฟรมใหม่หรือยัง
        self.connected = False     # ตอนนี้เชื่อมต่อกล้องอยู่ไหม
        self.running = False
        self.lock = threading.Lock()   # กุญแจ กัน 2 thread แย่งอ่าน/เขียนตัวแปร frame พร้อมกัน
        self.thread = None

    def start(self):
        """เริ่ม thread อ่านภาพ"""
        self.running = True
        # daemon=True = ถ้าโปรแกรมหลักปิด thread นี้ก็ปิดตามไปด้วย
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return self

    def _apply_esp32_settings(self):
        """
        ถ้าแหล่งภาพเป็น ESP32-CAM (ลิงก์ลงท้าย :81/stream) ส่งค่าขนาดภาพ/ความคมไปตั้งให้กล้อง
        เพราะกล้องลืมค่าทุกครั้งที่รีสตาร์ต — ทำตอนเชื่อมต่อทุกครั้ง จะได้ไม่ต้องตั้งเองบ่อย ๆ
        """
        parts = urlsplit(self.source)
        if parts.port != 81 or not parts.path.startswith("/stream"):
            return   # ไม่ใช่ ESP32-CAM (เช่น DroidCam) — ข้าม
        base = f"{parts.scheme}://{parts.hostname}"   # หน้าตั้งค่าอยู่ที่พอร์ต 80
        for var, val in config.ESP32_SETTINGS.items():
            try:
                requests.get(f"{base}/control", params={"var": var, "val": val}, timeout=3)
            except requests.RequestException:
                return   # กล้องยังไม่พร้อม — ไม่เป็นไร เดี๋ยวรอบเชื่อมใหม่จะลองอีก

    def _open(self):
        """เปิดการเชื่อมต่อกล้อง คืนค่า VideoCapture หรือ None ถ้าเปิดไม่ได้"""
        if isinstance(self.source, str):
            self._apply_esp32_settings()
            # กล้อง Wi-Fi (ESP32-CAM / มือถือ): ตั้งเวลารอสูงสุด ถ้าสัญญาณหาย cap.read() จะได้ไม่ค้างนาน
            # (ค่าเริ่มต้นของ OpenCV รอได้ ~30 วินาที ระหว่างนั้นโปรแกรมจะเหมือนภาพค้าง)
            timeout_ms = int(config.STREAM_TIMEOUT * 1000)
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
            ])
        else:
            cap = cv2.VideoCapture(self.source)   # webcam (เลข 0, 1, ...)
        if not cap.isOpened():
            cap.release()
            return None
        # ขอให้กล้องเก็บบัฟเฟอร์น้อยที่สุด (บางกล้องไม่รองรับ ก็ไม่เป็นไร)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader_loop(self):
        """วนอ่านภาพตลอดเวลา (ทำงานใน thread แยก)"""
        cap = None
        while self.running:
            # --- ยังไม่ได้เชื่อมต่อ → พยายามเปิดกล้อง ---
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.connected = False
                    print(f"[camera] เชื่อมต่อกล้องไม่ได้ ({self.source}) จะลองใหม่ใน {config.RECONNECT_DELAY} วินาที")
                    time.sleep(config.RECONNECT_DELAY)
                    continue
                print(f"[camera] เชื่อมต่อกล้องสำเร็จ: {self.source}")

            # --- อ่านภาพ 1 เฟรม ---
            try:
                ok, frame = cap.read()
            except cv2.error as e:
                # ภาพที่ส่งมาเสียกลางทาง (เจอบ่อยตอน Wi-Fi สะดุด) — ไม่ให้ thread ตาย ให้ไปเชื่อมใหม่แทน
                print(f"[camera] อ่านภาพผิดพลาด: {e}")
                ok, frame = False, None
            if not ok or frame is None:
                # อ่านไม่ได้ = กล้องหลุด → ปิดแล้วรอเชื่อมใหม่ (โปรแกรมหลักทำงานต่อ ไม่ปิดตัว)
                print("[camera] กล้องขาดการเชื่อมต่อ กำลังเชื่อมใหม่...")
                self.connected = False
                cap.release()
                cap = None
                time.sleep(config.RECONNECT_DELAY)
                continue

            # --- เก็บเป็นเฟรมล่าสุด (ทับของเก่า) ---
            with self.lock:
                self.frame = frame
                self.frame_id += 1
                self.connected = True

        if cap is not None:
            cap.release()

    def read(self):
        """
        ขอเฟรมล่าสุด
        คืนค่า (frame, frame_id) — frame เป็น None ถ้ายังไม่มีภาพ
        """
        with self.lock:
            if self.frame is None:
                return None, self.frame_id
            # .copy() เพื่อให้เอาไปวาดทับได้โดยไม่กระทบเฟรมต้นฉบับ
            return self.frame.copy(), self.frame_id

    def stop(self):
        """หยุดอ่านภาพ"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)


# ------------------------------------------------------------
# ทดสอบ: แสดงภาพสด + FPS  (กด q เพื่อออก)
# ------------------------------------------------------------
if __name__ == "__main__":
    cam = Camera().start()
    last_id = -1
    fps = 0.0
    prev_time = time.time()

    while True:
        frame, frame_id = cam.read()
        if frame is None or frame_id == last_id:
            # ยังไม่มีเฟรมใหม่ รอแป๊บเดียวแล้วลองใหม่
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            continue
        last_id = frame_id

        # คำนวณ FPS = 1 / เวลาที่ใช้ต่อเฟรม (เฉลี่ยให้ตัวเลขไม่กระโดด)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps:.1f}  ({w}x{h})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()
