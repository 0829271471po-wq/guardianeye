# -*- coding: utf-8 -*-
"""
person_tracker.py — ติดตามหลายคนในภาพ และผูก "ใบหน้า" เข้ากับ "ร่างกาย" ของแต่ละคน

ทำไมต้องมีไฟล์นี้:
  ถ้ามี 2 คนในภาพ แล้วคนหนึ่งล้ม เราต้องรู้ว่า "ร่างที่ล้ม" คือใคร
  ไม่ใช่แค่เอาชื่อล่าสุดที่เห็น (ซึ่งอาจเป็นอีกคนที่ยืนอยู่)

วิธีทำ:
1) แต่ละเฟรม MediaPipe ให้โครงร่างกายมาหลายชุด (คนละชุดต่อคน)
2) จับคู่โครงร่างกับ "คนเดิม" จากเฟรมก่อน ด้วยระยะห่างของจุดกึ่งกลางตัว (ใกล้สุด = คนเดิม)
   → ทุกคนมีหมายเลข (#1, #2, ...) และมี FallDetector ของตัวเอง
3) ตอนจดจำใบหน้าได้ ดูว่า "จมูก" ของคนไหนอยู่ในกรอบใบหน้านั้น → ติดชื่อให้คนนั้น
   ชื่อจะติดตัวไปตลอด แม้ตอนล้มหน้าจะไม่หันเข้ากล้องแล้วก็ตาม
4) คนไหนล้ม → รู้ชื่อจากตัวคนนั้นเลย
"""
import math

import config
from face_recognizer import UNKNOWN
from fall_detector import FallDetector, compute_metrics

NOSE = 0


def body_center(landmarks):
    """จุดกึ่งกลางของจุดข้อต่อที่มองเห็นชัด (ค่า 0-1) ใช้จับคู่คนเดิมข้ามเฟรม"""
    points = [p for p in landmarks if p.visibility >= config.LANDMARK_VISIBILITY] or list(landmarks)
    return (sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points))


def body_box(landmarks, frame_width, frame_height):
    """กรอบสี่เหลี่ยมครอบตัว (พิกเซล) จากจุดที่มองเห็นชัด"""
    points = [p for p in landmarks if p.visibility >= config.LANDMARK_VISIBILITY] or list(landmarks)
    xs = [p.x * frame_width for p in points]
    ys = [p.y * frame_height for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


class TrackedPerson:
    """ข้อมูลของคน 1 คนที่กำลังติดตาม"""

    def __init__(self, track_id, now):
        self.id = track_id
        self.detector = FallDetector()   # แต่ละคนมีตัวตรวจล้มของตัวเอง
        self.name = None                 # ชื่อที่จำได้จากใบหน้า (None = ยังไม่รู้)
        self.name_score = 0.0
        self.face_unknown = False        # เคยเห็นหน้าแล้วแต่ไม่รู้จัก
        self.landmarks = None
        self.center = None
        self.last_seen = now

    def label(self):
        if self.name:
            return self.name
        return UNKNOWN if self.face_unknown else "?"


class PersonTracker:
    def __init__(self):
        self.people = []      # list ของ TrackedPerson ที่ยังอยู่ในภาพ
        self.next_id = 1

    def update(self, all_landmarks, frame_width, frame_height, now):
        """
        รับโครงร่างทุกคนในเฟรมนี้ → จับคู่กับคนเดิม → อัปเดตตัวตรวจล้มของแต่ละคน
        คืนค่า list ของคนที่ "ควรส่งแจ้งเตือนล้ม" ในเฟรมนี้
        """
        centers = [body_center(lm) for lm in all_landmarks]

        # --- จับคู่แบบ "ใกล้สุดก่อน" (greedy): คำนวณระยะทุกคู่ แล้วเลือกคู่ที่ใกล้ที่สุดไล่ไป ---
        pairs = []
        for pi, person in enumerate(self.people):
            for ci, center in enumerate(centers):
                distance = math.dist(person.center, center)
                if distance <= config.TRACK_MATCH_DISTANCE:
                    pairs.append((distance, pi, ci))
        pairs.sort()
        matched_people, matched_poses = set(), set()
        assignment = {}   # ลำดับโครงร่าง → TrackedPerson
        for _, pi, ci in pairs:
            if pi in matched_people or ci in matched_poses:
                continue
            matched_people.add(pi)
            matched_poses.add(ci)
            assignment[ci] = self.people[pi]

        # --- โครงร่างที่ไม่ตรงกับใครเลย = คนใหม่เพิ่งเข้ามา ---
        for ci in range(len(all_landmarks)):
            if ci not in assignment:
                person = TrackedPerson(self.next_id, now)
                self.next_id += 1
                self.people.append(person)
                assignment[ci] = person
                print(f"[track] พบคนใหม่ #{person.id}")

        # --- อัปเดตตัวตรวจล้มของแต่ละคน ---
        to_alert = []
        for ci, person in assignment.items():
            landmarks = all_landmarks[ci]
            person.landmarks = landmarks
            person.center = centers[ci]
            person.last_seen = now
            metrics = compute_metrics(landmarks, frame_width, frame_height)
            if person.detector.update(metrics, now):
                to_alert.append(person)

        # คนที่ไม่เจอในเฟรมนี้ → บอกตัวตรวจล้มว่า "ไม่เห็น" และลบทิ้งถ้าหายไปนานเกิน
        seen = set(id(p) for p in assignment.values())
        for person in self.people:
            if id(person) not in seen:
                person.landmarks = None
                person.detector.update(None, now)
        lost = [p for p in self.people if now - p.last_seen > config.TRACK_LOST_SECONDS]
        for person in lost:
            print(f"[track] #{person.id} ({person.label()}) ออกจากภาพ")
        self.people = [p for p in self.people if p not in lost]
        return to_alert

    def assign_faces(self, face_results, frame_width, frame_height):
        """
        ผูกผลจดจำใบหน้ากับคน: ถ้าจมูกของใครอยู่ในกรอบใบหน้า (ขยายขอบเล็กน้อย) → ใบหน้านั้นเป็นของคนนั้น
        face_results = [((x, y, w, h), ชื่อ, คะแนน), ...] จาก FaceRecognizer.process()
        """
        for (x, y, w, h), name, score in face_results:
            margin_x, margin_y = w * 0.5, h * 0.5
            face_cx, face_cy = x + w / 2, y + h / 2
            best, best_distance = None, None
            for person in self.people:
                if person.landmarks is None:
                    continue
                nose = person.landmarks[NOSE]
                nx, ny = nose.x * frame_width, nose.y * frame_height
                inside = (x - margin_x <= nx <= x + w + margin_x) and (y - margin_y <= ny <= y + h + margin_y)
                if not inside:
                    continue
                distance = math.hypot(nx - face_cx, ny - face_cy)
                if best is None or distance < best_distance:
                    best, best_distance = person, distance
            if best is None:
                continue
            if name == UNKNOWN:
                best.face_unknown = True
            elif best.name != name or score > best.name_score:
                if best.name != name:
                    print(f"[track] #{best.id} คือ {name} (ความเหมือน {score:.2f})")
                best.name, best.name_score = name, score

    def name_for(self, person, faces=None, now=None):
        """
        ชื่อของคนที่ล้ม: ใช้ชื่อที่ติดตัวคนนั้นก่อน
        ถ้ายังไม่รู้ และในภาพมีอยู่คนเดียว → ใช้ชื่อล่าสุดที่ระบบจำได้ (แบบเดิม) แทน
        """
        if person.name:
            return person.name
        if faces is not None and len(self.people) == 1:
            return faces.recent_person(now)
        return None
