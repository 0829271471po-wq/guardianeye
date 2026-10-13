# -*- coding: utf-8 -*-
"""
notifier.py — ส่งแจ้งเตือนเข้า Telegram (ข้อความ + รูปภาพ)

- อ่าน TELEGRAM_BOT_TOKEN และ TELEGRAM_CHAT_ID จากไฟล์ .env (ไม่เขียนลงโค้ดตรง ๆ)
- ส่งใน "thread แยก" เพื่อให้หน้าจอภาพสดไม่ค้างระหว่างรออินเทอร์เน็ต
- ถ้าส่งไม่สำเร็จ จะลองใหม่ TELEGRAM_RETRIES ครั้ง แล้วพิมพ์ error ออกมา

ทดสอบไฟล์นี้อย่างเดียว (ส่งข้อความทดสอบ):  python notifier.py
"""
import os
import threading
import time

import cv2
import requests
from dotenv import load_dotenv

import config

# โหลดค่าจากไฟล์ .env ที่อยู่ในโฟลเดอร์โปรเจกต์ เข้าเป็น environment variable
load_dotenv(os.path.join(config.BASE_DIR, ".env"))
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


def is_configured():
    """ตั้งค่า Telegram ใน .env ครบหรือยัง"""
    return bool(BOT_TOKEN) and bool(CHAT_ID) and "ใส่" not in BOT_TOKEN


def _post_with_retry(method, data, files=None):
    """
    ส่งคำขอไปที่ Telegram Bot API พร้อมลองใหม่ถ้าล้มเหลว
    method เช่น "sendMessage" หรือ "sendPhoto"
    คืนค่า True ถ้าส่งสำเร็จ
    """
    for attempt in range(1, config.TELEGRAM_RETRIES + 1):
        try:
            response = requests.post(f"{API_URL}/{method}", data=data, files=files,
                                     timeout=config.TELEGRAM_TIMEOUT)
            if response.status_code == 200:
                print(f"[notifier] ส่ง Telegram สำเร็จ ({method})")
                return True
            # เซิร์ฟเวอร์ตอบกลับมาแต่ไม่สำเร็จ เช่น token ผิด / chat_id ผิด
            print(f"[notifier] Telegram ตอบกลับ error {response.status_code}: {response.text}")
        except requests.RequestException as error:
            # เน็ตหลุด / หมดเวลารอ
            print(f"[notifier] ส่งไม่สำเร็จ (ครั้งที่ {attempt}/{config.TELEGRAM_RETRIES}): {error}")
        time.sleep(config.TELEGRAM_RETRY_DELAY)
    print("[notifier] ERROR: ส่ง Telegram ไม่สำเร็จหลังลองครบทุกครั้ง")
    return False


def _send_photo_job(caption, image_bytes):
    """งานส่งรูป (ทำงานใน thread แยก)"""
    files = {"photo": ("snapshot.jpg", image_bytes, "image/jpeg")}
    _post_with_retry("sendPhoto", {"chat_id": CHAT_ID, "caption": caption}, files)


def _send_message_job(text):
    """งานส่งข้อความ (ทำงานใน thread แยก)"""
    _post_with_retry("sendMessage", {"chat_id": CHAT_ID, "text": text})


def send_photo(caption, frame):
    """
    ส่งรูป + ข้อความ เข้า Telegram แบบไม่รอ (เรียกแล้วกลับมาทำงานต่อทันที)
    frame = ภาพจาก OpenCV
    """
    if not is_configured():
        print("[notifier] ยังไม่ได้ตั้งค่า Telegram ใน .env — ข้ามการส่ง:", caption)
        return
    # แปลงภาพเป็นไฟล์ JPG ในหน่วยความจำ ก่อนส่ง
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        print("[notifier] แปลงรูปไม่สำเร็จ")
        return
    threading.Thread(target=_send_photo_job, args=(caption, buffer.tobytes()), daemon=True).start()


def send_message(text):
    """ส่งข้อความอย่างเดียวเข้า Telegram แบบไม่รอ"""
    if not is_configured():
        print("[notifier] ยังไม่ได้ตั้งค่า Telegram ใน .env — ข้ามการส่ง:", text)
        return
    threading.Thread(target=_send_message_job, args=(text,), daemon=True).start()


# ------------------------------------------------------------
# ทดสอบ: ส่งข้อความทดสอบเข้า Telegram
# ------------------------------------------------------------
if __name__ == "__main__":
    if not is_configured():
        print("ยังไม่ได้ใส่ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ในไฟล์ .env")
    else:
        # ทดสอบแบบรอผลเลย (ไม่ใช้ thread) จะได้เห็นว่าสำเร็จไหม
        _post_with_retry("sendMessage", {"chat_id": CHAT_ID, "text": "✅ GuardianEye: ทดสอบการแจ้งเตือนสำเร็จ"})
