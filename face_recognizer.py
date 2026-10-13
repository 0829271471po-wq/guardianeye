# -*- coding: utf-8 -*-
"""
face_recognizer.py — หาใบหน้า + บอกว่าเป็นใคร

ใช้โมเดลของ OpenCV 2 ตัว:
- YuNet  (cv2.FaceDetectorYN)  : หาตำแหน่งใบหน้าในภาพ
- SFace  (cv2.FaceRecognizerSF): แปลงใบหน้าเป็น "เวกเตอร์ตัวเลข 128 ค่า" (feature)
  ใบหน้าคนเดียวกัน → เวกเตอร์คล้ายกัน → cosine similarity สูง

ขั้นตอน:
1) ตอนเริ่มโปรแกรม โหลดรูปทุกคนใน known_faces/<ชื่อ>/ → คำนวณ feature เก็บไว้
2) ทุก N เฟรม หาใบหน้าในภาพ → เทียบ feature กับทุกคน → ได้ชื่อ หรือ "unknown"
3) จำชื่อล่าสุดที่เห็น (ใช้บอกว่าใครล้ม เพราะตอนล้มหน้ามักไม่หันเข้ากล้อง)
4) ถ้าเห็นคนแปลกหน้าต่อเนื่องนานพอ → แจ้งเตือน (มี cooldown)

ทดสอบไฟล์นี้อย่างเดียว:  python face_recognizer.py
"""
import os
import time

import cv2
import numpy as np

import config

UNKNOWN = "unknown"


def read_image(path):
    """อ่านรูป (รองรับที่อยู่ไฟล์ภาษาไทยบน Windows ซึ่ง cv2.imread อ่านไม่ได้)"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class FaceRecognizer:
    def __init__(self):
        # input size ตั้งไว้ก่อน เดี๋ยวจะปรับตามขนาดภาพจริงตอน detect
        self.detector = cv2.FaceDetectorYN.create(
            config.FACE_DETECTOR_MODEL, "", (320, 320), config.FACE_DETECT_SCORE, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(config.FACE_RECOGNIZER_MODEL, "")

        self.known = {}                 # {ชื่อ: [feature1, feature2, ...]}
        self.last_results = []          # ผลล่าสุด [(กรอบ x,y,w,h, ชื่อ, คะแนน), ...] ใช้วาดบนจอ

        # ความจำ "ใครอยู่ในภาพล่าสุด"
        self.last_known_name = None
        self.last_known_time = 0.0

        # ติดตามคนแปลกหน้า
        self.stranger_first_seen = None  # เริ่มเห็นคนแปลกหน้าตั้งแต่เมื่อไร
        self.stranger_last_seen = 0.0
        self.last_stranger_alert = -1e9

        self.load_known_faces()

    # ---------------- ตรวจจับใบหน้า ----------------
    def detect_faces(self, frame):
        """หาใบหน้าทั้งหมดในภาพ คืนค่า array (แต่ละแถวคือ 1 ใบหน้า: x, y, w, h, จุดตา/จมูก/ปาก, คะแนน)"""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15))

    def get_feature(self, frame, face):
        """ตัดภาพใบหน้าให้ตรง (align) แล้วแปลงเป็น feature"""
        aligned = self.recognizer.alignCrop(frame, face)
        return self.recognizer.feature(aligned)

    # ---------------- โหลดคนในบ้าน ----------------
    def load_known_faces(self):
        """อ่านรูปทุกคนใน known_faces/<ชื่อ>/*.jpg แล้วคำนวณ feature เก็บไว้"""
        self.known = {}
        if not os.path.isdir(config.KNOWN_FACES_DIR):
            print("[face] ยังไม่มีโฟลเดอร์ known_faces — ยังไม่มีใครลงทะเบียน")
            return
        for name in sorted(os.listdir(config.KNOWN_FACES_DIR)):
            person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
            if not os.path.isdir(person_dir):
                continue
            features = []
            for filename in os.listdir(person_dir):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                image = read_image(os.path.join(person_dir, filename))
                if image is None:
                    continue
                faces = self.detect_faces(image)
                if len(faces) == 0:
                    print(f"[face] ไม่พบใบหน้าในรูป {name}/{filename} — ข้าม")
                    continue
                # ถ้ามีหลายหน้า ใช้หน้าที่ใหญ่ที่สุด (w*h มากสุด)
                biggest = max(faces, key=lambda f: f[2] * f[3])
                features.append(self.get_feature(image, biggest))
            if features:
                self.known[name] = features
                print(f"[face] โหลด {name}: {len(features)} รูป")
        print(f"[face] ลงทะเบียนไว้ทั้งหมด {len(self.known)} คน")

    # ---------------- จดจำใบหน้า ----------------
    def identify(self, feature):
        """เทียบ feature กับทุกคน คืนค่า (ชื่อ, คะแนนความเหมือน)"""
        best_name, best_score = UNKNOWN, 0.0
        for name, features in self.known.items():
            for known_feature in features:
                # cosine similarity: 1 = เหมือนกันทุกอย่าง, 0 = ไม่เกี่ยวกันเลย
                score = self.recognizer.match(feature, known_feature, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_name, best_score = name, score
        if best_score < config.FACE_MATCH_THRESHOLD:
            return UNKNOWN, best_score
        return best_name, best_score

    def process(self, frame, now=None):
        """
        หาใบหน้า + จดจำ ในภาพนี้ (เรียกทุก N เฟรม)
        อัปเดตความจำชื่อล่าสุด และข้อมูลคนแปลกหน้า
        """
        if now is None:
            now = time.time()
        results = []
        saw_stranger = False
        for face in self.detect_faces(frame):
            name, score = self.identify(self.get_feature(frame, face))
            x, y, w, h = [int(v) for v in face[:4]]
            results.append(((x, y, w, h), name, score))
            if name == UNKNOWN:
                saw_stranger = True
            else:
                self.last_known_name = name
                self.last_known_time = now
        self.last_results = results

        # ติดตามว่าเห็นคนแปลกหน้า "ต่อเนื่อง" มานานแค่ไหน
        if saw_stranger:
            # ถ้าหายไปนานเกิน gap → นับว่าเป็นการเห็นครั้งใหม่ เริ่มนับเวลาใหม่
            if self.stranger_first_seen is None or now - self.stranger_last_seen > config.STRANGER_GAP_TOLERANCE:
                self.stranger_first_seen = now
            self.stranger_last_seen = now
        elif self.stranger_first_seen is not None and now - self.stranger_last_seen > config.STRANGER_GAP_TOLERANCE:
            self.stranger_first_seen = None
        return results

    def recent_person(self, now=None):
        """ชื่อคนที่เห็นล่าสุดภายใน IDENTITY_MEMORY_SECONDS (ไม่มี → None)"""
        if now is None:
            now = time.time()
        if self.last_known_name and now - self.last_known_time <= config.IDENTITY_MEMORY_SECONDS:
            return self.last_known_name
        return None

    def should_alert_stranger(self, now=None):
        """คืนค่า True ถ้าเห็นคนแปลกหน้าต่อเนื่องครบเวลา และพ้น cooldown แล้ว"""
        if now is None:
            now = time.time()
        if self.stranger_first_seen is None:
            return False
        long_enough = now - self.stranger_first_seen >= config.STRANGER_SECONDS
        cooldown_over = now - self.last_stranger_alert >= config.STRANGER_COOLDOWN
        if long_enough and cooldown_over:
            self.last_stranger_alert = now
            return True
        return False

    def draw(self, frame):
        """วาดกรอบใบหน้า + ชื่อ (เขียว = คนในบ้าน, แดง = ไม่รู้จัก)"""
        for (x, y, w, h), name, score in self.last_results:
            color = (0, 0, 255) if name == UNKNOWN else (0, 200, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def display_name(name):
    """แปลงชื่อโฟลเดอร์ (อังกฤษ) เป็นชื่อสำหรับข้อความ เช่น grandma → คุณยาย"""
    if not name:
        return config.UNKNOWN_PERSON_TEXT
    return config.PERSON_DISPLAY_NAMES.get(name, name)


# ------------------------------------------------------------
# ทดสอบ: แสดงกรอบใบหน้า + ชื่อ  (กด q เพื่อออก)
# ------------------------------------------------------------
if __name__ == "__main__":
    from camera import Camera

    cam = Camera().start()
    faces = FaceRecognizer()
    last_id = -1
    while True:
        frame, frame_id = cam.read()
        if frame is None or frame_id == last_id:
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            continue
        last_id = frame_id
        faces.process(frame)
        faces.draw(frame)
        if faces.should_alert_stranger():
            print(">>> (ทดสอบ) ตรงนี้จะแจ้งเตือนคนแปลกหน้า")
        cv2.imshow("Face test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.stop()
    cv2.destroyAllWindows()
