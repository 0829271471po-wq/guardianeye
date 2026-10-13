# -*- coding: utf-8 -*-
"""
telegram_bot.py — ให้ถามบอทใน Telegram ได้ว่า "ตอนนี้ในห้องมีคนกี่คน"

วิธีทำงาน:
1) main.py อัปเดต "สถานะห้องล่าสุด" (จำนวนคน + ชื่อ + ภาพ) ลงใน RoomStatus ทุกเฟรม
2) CommandListener ทำงานใน thread แยก คอยถาม Telegram ว่ามีข้อความใหม่ไหม (long polling)
3) ถ้าข้อความมีคำว่า "กี่คน" / "มีคน" / "ใครอยู่" หรือคำสั่ง /count → ตอบจำนวนคน + ชื่อ + รูปห้องตอนนี้

ทำไมต้องรวมอยู่ใน main.py: ESP32-CAM ส่งภาพได้ทีละโปรแกรม จึงเปิดโปรแกรมแยกมาดูกล้องไม่ได้
ความปลอดภัย: ตอบเฉพาะแชตที่ตรงกับ TELEGRAM_CHAT_ID ใน .env เท่านั้น คนอื่นทักมาจะไม่ตอบ
"""
import threading
import time
from datetime import datetime

import requests

import config
import notifier
from face_recognizer import display_name

# คำที่ถือว่าเป็นคำถาม "มีคนกี่คน"
COUNT_KEYWORDS = ("กี่คน", "มีคน", "ใครอยู่", "ใครบ้าง", "/count", "/people", "/status")

HELP_TEXT = ("🤖 GuardianEye พร้อมตอบคำถาม\n"
             f"ลองพิมพ์: \"{config.ROOM_NAME}มีคนกี่คน\" หรือ /count")

POLL_SECONDS = 25        # รอข้อความใหม่จาก Telegram นานสุดกี่วินาทีต่อรอบ (long polling)
STALE_SECONDS = 5        # ข้อมูลจากกล้องเก่ากว่านี้ = ระบบภาพค้าง/กล้องหลุด


class RoomStatus:
    """สถานะห้องล่าสุด ใช้ร่วมกันระหว่าง main.py (เขียน) กับบอท (อ่าน) จึงต้องมี lock"""

    def __init__(self):
        self._lock = threading.Lock()
        self._names = []          # ชื่อของแต่ละคนในภาพ (None = ยังไม่รู้ว่าใคร)
        self._unknown = []        # True = เห็นหน้าแล้วแต่ไม่รู้จัก (คนแปลกหน้า)
        self._frame = None
        self._time = 0.0

    def update(self, tracker, frame):
        """เรียกจาก main.py ทุกเฟรม หลังวาดผลบนภาพแล้ว"""
        with self._lock:
            self._names = [p.name for p in tracker.people]
            self._unknown = [p.face_unknown for p in tracker.people]
            self._frame = frame.copy()
            self._time = time.time()

    def snapshot(self):
        """คืนค่า (ชื่อ, คนแปลกหน้า, ภาพ, อายุข้อมูลเป็นวินาที)"""
        with self._lock:
            return list(self._names), list(self._unknown), self._frame, time.time() - self._time


def build_answer(names, unknown):
    """แต่งข้อความคำตอบจากรายชื่อคนในภาพ"""
    time_text = datetime.now().strftime("%H:%M")
    if not names:
        return f"🏠 {config.ROOM_NAME}ตอนนี้ไม่มีคน (เวลา {time_text})"
    lines = [f"🏠 {config.ROOM_NAME}ตอนนี้มี {len(names)} คน (เวลา {time_text})"]
    for name, is_stranger in zip(names, unknown):
        if name:
            lines.append(f"• {display_name(name)}")
        elif is_stranger:
            lines.append("• คนแปลกหน้า ⚠️")
        else:
            lines.append("• ยังไม่เห็นหน้า (ไม่รู้ว่าใคร)")
    return "\n".join(lines)


class CommandListener:
    """ฟังข้อความจาก Telegram ใน thread แยก แล้วตอบคำถามเรื่องจำนวนคน"""

    def __init__(self, status):
        self.status = status
        self.offset = None        # หมายเลขข้อความถัดไปที่ยังไม่ได้อ่าน
        self.running = False

    def start(self):
        if not notifier.is_configured():
            print("[bot] ยังไม่ได้ตั้งค่า Telegram ใน .env — ปิดบอทตอบคำถาม")
            return self
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("[bot] พร้อมตอบคำถามใน Telegram แล้ว (เช่น \"มีคนกี่คน\")")
        return self

    def stop(self):
        self.running = False

    def _get_updates(self, timeout):
        params = {"timeout": timeout, "allowed_updates": '["message"]'}
        if self.offset is not None:
            params["offset"] = self.offset
        response = requests.get(f"{notifier.API_URL}/getUpdates", params=params,
                                timeout=timeout + 10)
        response.raise_for_status()
        return response.json().get("result", [])

    def _loop(self):
        # ข้ามข้อความเก่าที่ส่งมาตอนโปรแกรมยังไม่ได้เปิด (จะได้ไม่ตอบคำถามที่ค้างจากเมื่อวาน)
        try:
            old = self._get_updates(0)
            if old:
                self.offset = old[-1]["update_id"] + 1
        except requests.RequestException as error:
            print("[bot] เชื่อมต่อ Telegram ไม่ได้:", error)

        while self.running:
            try:
                updates = self._get_updates(POLL_SECONDS)
            except requests.RequestException as error:
                print("[bot] รับข้อความไม่สำเร็จ จะลองใหม่:", error)
                time.sleep(5)
                continue
            for update in updates:
                self.offset = update["update_id"] + 1
                self._handle(update.get("message") or {})

    def _handle(self, message):
        text = (message.get("text") or "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        if not text:
            return
        if chat_id != notifier.CHAT_ID:
            print(f"[bot] มีแชตอื่น ({chat_id}) ทักมา — ไม่ตอบ")
            return

        if not any(word in text.lower() for word in COUNT_KEYWORDS):
            notifier.send_message(HELP_TEXT)
            return

        names, unknown, frame, age = self.status.snapshot()
        if frame is None or age > STALE_SECONDS:
            notifier.send_message("📷 ตอนนี้กล้องขาดการเชื่อมต่อ ยังนับคนไม่ได้ กรุณาลองใหม่อีกครั้ง")
            return
        answer = build_answer(names, unknown)
        print("[bot] ตอบคำถาม:", answer.splitlines()[0])
        notifier.send_photo(answer, frame)


# ------------------------------------------------------------
# ทดสอบ: ดูหน้าตาข้อความคำตอบ (ไม่ต่อกล้อง ไม่ส่ง Telegram)
# ------------------------------------------------------------
if __name__ == "__main__":
    print(build_answer([], []))
    print()
    print(build_answer(["me", None, None], [False, True, False]))
