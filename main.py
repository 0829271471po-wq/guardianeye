# -*- coding: utf-8 -*-
"""
main.py — จุดเริ่มต้นของ GuardianEye

ลำดับการทำงานในแต่ละรอบ (loop):
  1. อ่านเฟรมล่าสุดจากกล้อง                     (camera.py)
  2. หาโครงร่างกายทุกคน → ติดตามแต่ละคน → ตรวจการล้มแยกรายคน (fall_detector.py, person_tracker.py)
  3. ทุก N เฟรม: หาใบหน้า → จดจำว่าเป็นใคร → ติดชื่อให้ร่างของคนนั้น (face_recognizer.py)
  4. ถ้ายืนยันว่าล้ม (บอกชื่อคนที่ล้ม) / เจอคนแปลกหน้า → บันทึก + แจ้งเตือน (event_logger.py, notifier.py)
  5. วาดผลลัพธ์บนจอ + เก็บสถานะห้องไว้ให้บอท Telegram ตอบคำถาม "มีคนกี่คน" (telegram_bot.py)

รัน:  python main.py    (กด q เพื่อออก)
"""
import time
from datetime import datetime

import cv2
import numpy as np

import config
import event_logger
import notifier
from camera import Camera
from face_recognizer import FaceRecognizer, display_name
from fall_detector import (FALL_CONFIRMED, FALL_SUSPECTED, NORMAL,
                           PoseEstimator, draw_skeleton)
from person_tracker import PersonTracker, body_box
from telegram_bot import CommandListener, RoomStatus

# สีของแต่ละสถานะ (OpenCV ใช้ลำดับสี B, G, R)
STATE_COLORS = {
    NORMAL: (0, 200, 0),            # เขียว
    FALL_SUSPECTED: (0, 220, 255),  # เหลือง
    FALL_CONFIRMED: (0, 0, 255),    # แดง
}


def put_text(frame, text, y, color=(255, 255, 255), scale=0.7):
    """เขียนข้อความมุมบนซ้าย มีขอบดำรอบตัวอักษรให้อ่านง่ายบนทุกพื้นหลัง"""
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# ลำดับความรุนแรง ใช้เลือกสถานะที่ "แย่ที่สุด" มาแสดงมุมจอ
SEVERITY = {NORMAL: 0, FALL_SUSPECTED: 1, FALL_CONFIRMED: 2}


def draw_people(frame, tracker):
    """วาดกรอบรอบตัวแต่ละคน + ป้าย "#หมายเลข ชื่อ สถานะ" (สีตามสถานะ)"""
    h, w = frame.shape[:2]
    for person in tracker.people:
        if person.landmarks is None:
            continue
        state = person.detector.state
        color = STATE_COLORS[state]
        x1, y1, x2, y2 = body_box(person.landmarks, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{person.id} {person.label()}"
        if state != NORMAL:
            label += f" {state}"
        left = person.detector.seconds_left()
        if left is not None:
            label += f" {left:.0f}s"
        y = max(y1 - 10, 20)
        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_status(frame, tracker, fps):
    """วาดแผงสถานะมุมบนซ้าย: สถานะที่แย่ที่สุด, จำนวนคน, FPS"""
    states = [p.detector.state for p in tracker.people] or [NORMAL]
    worst = max(states, key=SEVERITY.get)
    put_text(frame, f"Status: {worst}", 30, STATE_COLORS[worst], 0.8)
    put_text(frame, f"People: {len(tracker.people)}", 60)
    put_text(frame, f"FPS: {fps:.1f}", 90)


def disconnected_screen():
    """ภาพสีดำพร้อมข้อความ ใช้แสดงตอนกล้องขาดการเชื่อมต่อ"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    put_text(frame, "Camera disconnected - reconnecting...", 240, (0, 0, 255))
    return frame


def handle_fall(frame, name):
    """ยืนยันว่าล้มแล้ว → บันทึก + ส่ง Telegram (name = ชื่อโฟลเดอร์ของคนที่ล้ม หรือ None ถ้าไม่รู้)"""
    person = display_name(name)
    time_text = datetime.now().strftime("%H:%M")
    caption = (f"⚠️ GuardianEye: ตรวจพบ {person} ล้มที่{config.ROOM_NAME}! "
               f"เวลา {time_text} กรุณาตรวจสอบด่วน")
    event_logger.log_event("fall", person, frame)
    notifier.send_photo(caption, frame)


def handle_stranger(frame):
    """เห็นคนแปลกหน้านานพอ → บันทึก + ส่ง Telegram"""
    time_text = datetime.now().strftime("%H:%M")
    caption = f"👤 GuardianEye: พบบุคคลที่ไม่รู้จักใน{config.ROOM_NAME} เวลา {time_text}"
    event_logger.log_event("stranger", "ไม่รู้จัก", frame)
    notifier.send_photo(caption, frame)


def main():
    print("กำลังเริ่ม GuardianEye...")
    if not notifier.is_configured():
        print("คำเตือน: ยังไม่ได้ตั้งค่า Telegram ใน .env — ระบบจะทำงานแต่ไม่ส่งแจ้งเตือน")

    cam = Camera().start()
    pose = PoseEstimator()
    tracker = PersonTracker()
    room_status = RoomStatus()                       # ให้บอท Telegram ตอบได้ว่าในห้องมีกี่คน
    bot = CommandListener(room_status).start()

    # ถ้าไม่มีไฟล์โมเดลใบหน้า ให้ระบบตรวจล้มทำงานต่อได้ (แค่ไม่มีการจดจำใบหน้า)
    try:
        faces = FaceRecognizer()
    except cv2.error as error:
        print("[main] โหลดโมเดลใบหน้าไม่ได้ — ปิดการจดจำใบหน้าไว้ก่อน:", error)
        faces = None

    last_id = -1
    frame_count = 0
    fps = 0.0
    prev_time = time.time()

    while True:
        frame, frame_id = cam.read()

        # ---------- กล้องหลุด ----------
        if not cam.connected:
            cv2.imshow(config.WINDOW_NAME, disconnected_screen())
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            continue

        # ---------- ยังไม่มีเฟรมใหม่ ----------
        if frame is None or frame_id == last_id:
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
            continue
        last_id = frame_id
        frame_count += 1
        now = time.time()
        h, w = frame.shape[:2]

        # ---------- 1) หาโครงร่างทุกคน → ติดตาม → ตรวจล้มแยกรายคน ----------
        all_landmarks = pose.detect_all(frame)
        fallen_people = tracker.update(all_landmarks, w, h, now)

        # ---------- 2) จดจำใบหน้า (ทุก N เฟรม) → ติดชื่อให้ร่างของคนนั้น ----------
        if faces is not None and frame_count % config.FACE_EVERY_N_FRAMES == 0:
            face_results = faces.process(frame, now)
            tracker.assign_faces(face_results, w, h)

        # ---------- 3) วาดผลบนจอ ----------
        if config.SHOW_SKELETON:
            for landmarks in all_landmarks:
                draw_skeleton(frame, landmarks)
        if faces is not None:
            faces.draw(frame)
        draw_people(frame, tracker)

        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        draw_status(frame, tracker, fps)

        # ---------- 4) แจ้งเตือน (ใช้ภาพที่วาดแล้ว จะได้เห็นว่าใครล้มในรูปที่ส่ง) ----------
        for person in fallen_people:
            handle_fall(frame, tracker.name_for(person, faces, now))
        if faces is not None and faces.should_alert_stranger(now):
            handle_stranger(frame)

        room_status.update(tracker, frame)
        cv2.imshow(config.WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("กำลังปิดโปรแกรม...")
    bot.stop()
    pose.close()
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
