# -*- coding: utf-8 -*-
"""
register_face.py — ถ่ายรูปลงทะเบียนใบหน้าคนในบ้าน

วิธีใช้:  python register_face.py grandma
  - กด SPACE (เว้นวรรค) = ถ่ายรูป (ถ่ายได้เฉพาะตอนเห็นใบหน้า 1 หน้าพอดี)
  - กด q = เสร็จสิ้น
แนะนำถ่าย 5-10 รูป หลายมุม (ตรง / เอียงซ้าย / เอียงขวา / ก้มเล็กน้อย / เงยเล็กน้อย)
รูปจะเก็บไว้ที่ known_faces/<ชื่อ>/

ตั้งชื่อเป็นภาษาอังกฤษ แล้วไปจับคู่ชื่อไทยใน config.py → PERSON_DISPLAY_NAMES
"""
import os
import sys
from datetime import datetime

import cv2

import config
from camera import Camera
from face_recognizer import FaceRecognizer

TARGET_PHOTOS = 10


def main():
    if len(sys.argv) < 2:
        print("วิธีใช้: python register_face.py <ชื่อภาษาอังกฤษ>   เช่น  python register_face.py grandma")
        return
    name = sys.argv[1]
    save_dir = os.path.join(config.KNOWN_FACES_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    cam = Camera().start()
    face_tool = FaceRecognizer()   # ใช้แค่ตัวตรวจจับใบหน้า (YuNet)
    count = len(os.listdir(save_dir))  # นับรูปที่มีอยู่แล้ว (ถ้าเคยลงทะเบียนไว้)
    print(f"ลงทะเบียน '{name}' — กด SPACE เพื่อถ่าย, q เพื่อจบ")

    while True:
        frame, _ = cam.read()
        if frame is None:
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        faces = face_tool.detect_faces(frame)
        display = frame.copy()     # วาดบนสำเนา จะได้บันทึกรูปต้นฉบับที่ไม่มีกรอบ
        for face in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(faces) == 1:
            status, color = "OK - press SPACE", (0, 255, 0)
        elif len(faces) == 0:
            status, color = "No face", (0, 0, 255)
        else:
            status, color = "Only 1 face please", (0, 0, 255)
        cv2.putText(display, f"{name}: {count}/{TARGET_PHOTOS}  {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Register face", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and len(faces) == 1:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                buffer.tofile(os.path.join(save_dir, filename))
                count += 1
                print(f"บันทึกรูปที่ {count}: {filename}")

    cam.stop()
    cv2.destroyAllWindows()
    print(f"เสร็จแล้ว — {name} มีรูปทั้งหมด {count} รูป")


if __name__ == "__main__":
    main()
