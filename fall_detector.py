# -*- coding: utf-8 -*-
"""
fall_detector.py — ตรวจจับการล้มด้วย Pose Estimation + State Machine

มี 3 ส่วน:
1) PoseEstimator  : ใช้ MediaPipe หาจุดข้อต่อร่างกาย 33 จุดจากภาพ
2) compute_metrics: คำนวณ มุมลำตัว / ตำแหน่งสะโพก / อัตราส่วนกรอบร่างกาย
3) FallDetector   : State Machine ตัดสินว่า ปกติ / สงสัยว่าล้ม / ล้มจริง

หมายเหตุเวอร์ชัน: ใช้ MediaPipe "Tasks API" (PoseLandmarker)
เพราะ API แบบเก่า (mp.solutions.pose) ถูกถอดออกใน mediapipe เวอร์ชันใหม่แล้ว
ต้องมีไฟล์โมเดล models/pose_landmarker_lite.task (รัน python download_models.py)

ทดสอบไฟล์นี้อย่างเดียว:  python fall_detector.py
"""
import math
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import config

# หมายเลขจุดข้อต่อของ MediaPipe Pose ที่เราใช้
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

# เส้นเชื่อมจุดข้อต่อ สำหรับวาดโครงร่างกาย (คู่หมายเลขจุด)
SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),        # ลำตัว
    (11, 13), (13, 15), (12, 14), (14, 16),        # แขน
    (23, 25), (25, 27), (24, 26), (26, 28),        # ขา
    (27, 31), (28, 32),                            # เท้า
    (0, 11), (0, 12),                              # หัว → ไหล่
]

# ชื่อสถานะ
NORMAL = "NORMAL"
FALL_SUSPECTED = "FALL_SUSPECTED"
FALL_CONFIRMED = "FALL_CONFIRMED"


# ============================================================
# 1) PoseEstimator — หาจุดข้อต่อร่างกาย
# ============================================================
class PoseEstimator:
    def __init__(self, model_path=config.POSE_MODEL_PATH):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,   # โหมดวิดีโอ = ใช้ข้อมูลเฟรมก่อนหน้าช่วยให้ติดตามลื่นขึ้น
            num_poses=config.MAX_PEOPLE,             # ตรวจได้หลายคนพร้อมกัน (แยกว่าใครล้มใน person_tracker.py)
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def detect_all(self, frame_bgr):
        """
        รับภาพ (BGR จาก OpenCV) คืนค่า list ของโครงร่าง "ทุกคน" ในภาพ (ไม่เจอใคร = list ว่าง)
        โครงร่างแต่ละคน = จุดข้อต่อ 33 จุด แต่ละจุดมี .x .y (0-1 เทียบกับขนาดภาพ) และ .visibility (ความมั่นใจ)
        """
        # MediaPipe ต้องการภาพสี RGB แต่ OpenCV ใช้ BGR → แปลงก่อน
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # โหมด VIDEO ต้องส่งเวลา (มิลลิวินาที) ที่ "เพิ่มขึ้นเรื่อย ๆ" เสมอ
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.pose_landmarks or []

    def detect(self, frame_bgr):
        """เหมือน detect_all แต่คืนแค่คนแรก (ไม่เจอคน → None) ใช้ในโหมดทดสอบคนเดียว"""
        people = self.detect_all(frame_bgr)
        return people[0] if people else None

    def close(self):
        self.landmarker.close()


# ============================================================
# 2) คำนวณค่าจากจุดข้อต่อ
# ============================================================
def compute_metrics(landmarks, frame_width, frame_height):
    """
    คำนวณค่าที่ใช้ตัดสินการล้ม คืนค่า dict:
      torso_angle : มุมลำตัวเทียบแนวตั้ง (องศา) ยืน ≈ 0, นอน ≈ 90
      hip_y       : ตำแหน่งแนวตั้งของสะโพก (0 = บนสุดของภาพ, 1 = ล่างสุด)
      aspect_ratio: กว้าง/สูง ของกรอบที่ครอบตัว (>1 = ตัวอยู่แนวนอน)
    ถ้าจุดไหล่/สะโพกมองไม่ชัด คืนค่า None
    """
    key_points = [landmarks[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)]
    if any(p.visibility < config.LANDMARK_VISIBILITY for p in key_points):
        return None

    # จุดกึ่งกลางไหล่ และจุดกึ่งกลางสะโพก (ค่า 0-1)
    shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2

    # --- มุมลำตัว ---
    # แปลงเป็นพิกเซลก่อน เพราะภาพกว้างไม่เท่าสูง (640x480) ถ้าใช้ค่า 0-1 ตรง ๆ มุมจะเพี้ยน
    dx = (shoulder_x - hip_x) * frame_width     # ระยะแนวนอน
    dy = (shoulder_y - hip_y) * frame_height    # ระยะแนวตั้ง
    # atan2(แนวนอน, แนวตั้ง) = มุมเทียบกับเส้นตั้ง → 0° = ตั้งตรง, 90° = นอนราบ
    torso_angle = math.degrees(math.atan2(abs(dx), abs(dy)))

    # --- อัตราส่วนกรอบร่างกาย (ใช้เฉพาะจุดที่มองเห็นชัด) ---
    xs = [p.x * frame_width for p in landmarks if p.visibility >= config.LANDMARK_VISIBILITY]
    ys = [p.y * frame_height for p in landmarks if p.visibility >= config.LANDMARK_VISIBILITY]
    box_w = max(xs) - min(xs)
    box_h = max(ys) - min(ys)
    aspect_ratio = box_w / box_h if box_h > 0 else 0.0

    return {
        "torso_angle": torso_angle,
        "hip_y": hip_y,
        "aspect_ratio": aspect_ratio,
    }


def draw_skeleton(frame, landmarks):
    """วาดโครงร่างกาย (เส้นสีฟ้า + จุดสีขาว) ลงบนภาพ"""
    h, w = frame.shape[:2]

    def to_pixel(i):
        return int(landmarks[i].x * w), int(landmarks[i].y * h)

    for a, b in SKELETON_CONNECTIONS:
        if landmarks[a].visibility > config.LANDMARK_VISIBILITY and landmarks[b].visibility > config.LANDMARK_VISIBILITY:
            cv2.line(frame, to_pixel(a), to_pixel(b), (255, 200, 0), 2)
    for i, p in enumerate(landmarks):
        if p.visibility > config.LANDMARK_VISIBILITY:
            cv2.circle(frame, to_pixel(i), 3, (255, 255, 255), -1)


# ============================================================
# 3) FallDetector — State Machine
# ============================================================
class FallDetector:
    """
    สถานะ:
      NORMAL          ปกติ
      FALL_SUSPECTED  สะโพกตกเร็ว + ตัวอยู่แนวนอน → สงสัยว่าล้ม เริ่มจับเวลา
      FALL_CONFIRMED  นอนอยู่ครบ LYING_SECONDS → ยืนยันว่าล้ม แจ้งเตือน

    การเปลี่ยนสถานะ:
      NORMAL         → FALL_SUSPECTED : ตกเร็ว และ (มุม > 60° หรือ กว้าง/สูง > 1)
      FALL_SUSPECTED → NORMAL         : ลุกขึ้นได้ก่อนครบเวลา (เช่น แค่ก้มเก็บของ)
      FALL_SUSPECTED → FALL_CONFIRMED : ยังนอนอยู่ครบ LYING_SECONDS
      FALL_CONFIRMED → NORMAL         : ลุกขึ้นแล้ว
    """

    def __init__(self):
        self.state = NORMAL
        self.hip_history = deque()       # เก็บ (เวลา, hip_y) ย้อนหลัง ใช้คำนวณความเร็วการตก
        self.suspect_start = None        # เวลาที่เริ่มสงสัยว่าล้ม
        self.upright_since = None        # เวลาที่เริ่มกลับมาตั้งตรง (ใช้เช็กว่าลุกจริง)
        self.last_seen_person = None     # เวลาล่าสุดที่เห็นคน
        self.last_alert_time = -1e9      # เวลาที่ส่งเตือนครั้งล่าสุด (ค่าติดลบมาก = ยังไม่เคยส่ง)
        self.last_metrics = None         # ค่าล่าสุด เอาไว้แสดงบนจอ
        self.hip_drop = 0.0              # สะโพกตกไปเท่าไรในช่วง VELOCITY_WINDOW
        self.last_fast_drop = -1e9       # เวลาล่าสุดที่ตรวจพบว่าสะโพกตกเร็ว

    def _is_horizontal(self, m):
        """ตัวอยู่ในแนวนอนหรือไม่"""
        return m["torso_angle"] > config.TORSO_ANGLE_THRESHOLD or m["aspect_ratio"] > config.ASPECT_RATIO_THRESHOLD

    def _update_hip_drop(self, now, hip_y):
        """
        คำนวณว่าสะโพก "ตกลง" ไปเท่าไรในช่วง VELOCITY_WINDOW วินาทีล่าสุด
        y มากขึ้น = ต่ำลงในภาพ ดังนั้น ตกลง = hip_y ปัจจุบัน - hip_y ที่สูงที่สุดในช่วงนั้น
        """
        self.hip_history.append((now, hip_y))
        # ทิ้งข้อมูลที่เก่ากว่าช่วงเวลาที่สนใจ
        while self.hip_history and now - self.hip_history[0][0] > config.VELOCITY_WINDOW:
            self.hip_history.popleft()
        highest = min(y for _, y in self.hip_history)
        self.hip_drop = hip_y - highest

    def update(self, metrics, now=None):
        """
        เรียกทุกเฟรม ส่ง metrics จาก compute_metrics() (หรือ None ถ้าไม่เจอคน)
        คืนค่า True เฉพาะ "จังหวะที่ควรส่งแจ้งเตือนล้ม" (ครั้งเดียวต่อการล้ม และพ้น cooldown แล้ว)
        """
        if now is None:
            now = time.time()

        # ---------- กรณีไม่เจอคนในเฟรม ----------
        if metrics is None:
            # pose อาจหลุดชั่วคราว → ยังไม่เปลี่ยนสถานะ ถ้าหายไปไม่นาน
            if self.last_seen_person is not None and now - self.last_seen_person > config.NO_PERSON_TOLERANCE:
                # หายไปนานแล้ว → ล้างข้อมูล กลับสู่ปกติ (เช่น คนเดินออกจากกล้อง)
                self.hip_history.clear()
                if self.state == FALL_SUSPECTED:
                    self.state = NORMAL
                    self.suspect_start = None
            return False

        # ---------- กรณีเจอคน ----------
        self.last_seen_person = now
        self.last_metrics = metrics
        self._update_hip_drop(now, metrics["hip_y"])
        horizontal = self._is_horizontal(metrics)

        # จับเวลาว่ากลับมาตั้งตรงต่อเนื่องนานแค่ไหน
        if horizontal:
            self.upright_since = None
        elif self.upright_since is None:
            self.upright_since = now
        recovered = self.upright_since is not None and now - self.upright_since >= config.RECOVER_SECONDS

        should_alert = False

        # จำเวลาที่สะโพกตกเร็วไว้ เพราะตอนล้มจริง "ตกเร็ว" กับ "นอนราบ" อาจไม่เกิดในเฟรมเดียวกัน
        if self.hip_drop > config.FALL_VELOCITY_THRESHOLD:
            self.last_fast_drop = now
        fell_fast_recently = now - self.last_fast_drop <= config.FALL_TO_LYING_WINDOW

        if self.state == NORMAL:
            if fell_fast_recently and horizontal:
                self.state = FALL_SUSPECTED
                self.suspect_start = now
                print("[fall] สงสัยว่าล้ม — เริ่มจับเวลา")

        elif self.state == FALL_SUSPECTED:
            if recovered:
                self.state = NORMAL
                self.suspect_start = None
                print("[fall] ลุกขึ้นได้ — ไม่ใช่การล้ม")
            elif now - self.suspect_start >= config.LYING_SECONDS:
                self.state = FALL_CONFIRMED
                print("[fall] ยืนยันว่าล้ม!")
                # ส่งเตือนเฉพาะถ้าพ้นช่วง cooldown
                if now - self.last_alert_time >= config.ALERT_COOLDOWN:
                    self.last_alert_time = now
                    should_alert = True

        elif self.state == FALL_CONFIRMED:
            if recovered:
                self.state = NORMAL
                self.suspect_start = None
                print("[fall] ลุกขึ้นแล้ว — กลับสู่ปกติ")

        return should_alert

    def seconds_left(self, now=None):
        """ตอน FALL_SUSPECTED เหลืออีกกี่วินาทีจะยืนยันว่าล้ม (ใช้แสดงนับถอยหลัง)"""
        if self.state != FALL_SUSPECTED or self.suspect_start is None:
            return None
        if now is None:
            now = time.time()
        return max(0.0, config.LYING_SECONDS - (now - self.suspect_start))


# ------------------------------------------------------------
# ทดสอบ: แสดงโครงร่างกาย + มุมลำตัว + สถานะ  (กด q เพื่อออก)
# ------------------------------------------------------------
if __name__ == "__main__":
    from camera import Camera

    cam = Camera().start()
    pose = PoseEstimator()
    detector = FallDetector()
    last_id = -1
    colors = {NORMAL: (0, 200, 0), FALL_SUSPECTED: (0, 220, 255), FALL_CONFIRMED: (0, 0, 255)}

    while True:
        frame, frame_id = cam.read()
        if frame is None or frame_id == last_id:
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            continue
        last_id = frame_id

        h, w = frame.shape[:2]
        landmarks = pose.detect(frame)
        metrics = compute_metrics(landmarks, w, h) if landmarks else None
        if detector.update(metrics):
            print(">>> (ทดสอบ) ตรงนี้จะส่งแจ้งเตือน Telegram")

        if landmarks:
            draw_skeleton(frame, landmarks)
        cv2.putText(frame, detector.state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[detector.state], 2)
        if metrics:
            text = f"angle {metrics['torso_angle']:.0f}  ratio {metrics['aspect_ratio']:.2f}  drop {detector.hip_drop:.2f}"
            cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        left = detector.seconds_left()
        if left is not None:
            cv2.putText(frame, f"confirm in {left:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        cv2.imshow("Fall detector test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pose.close()
    cam.stop()
    cv2.destroyAllWindows()
