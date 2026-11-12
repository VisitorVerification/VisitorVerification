from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.types import PersonBox, RecognitionResult
from config import FONT_PATH, WINDOW_NAME

# 배너 모양 (10장: 배너 색·글자 크기는 B가 정함)
BANNER_RATIO = 0.18          # 배너 높이 = 프레임 높이의 18% (5.3절)
BANNER_ALPHA = 0.72          # 반투명 정도 (1이면 불투명)
TEXT_RATIO = 0.45            # 최대 글자 크기 = 배너 높이의 45%. 폭을 넘으면 자동으로 줄임
TEXT_MARGIN = 24             # 글자 좌우 여백 (픽셀)
BANNER_COLORS = {            # BGR
    "welcome": (67, 160, 46),     # 초록: 등록자
    "counter": (47, 47, 211),     # 빨강: 미등록자, 인식 불가
    "hint": (97, 97, 97),         # 회색: 얼굴 안내
}

DEBUG_PERSON = (0, 200, 0)
DEBUG_ANIMAL = (160, 160, 160)
DEBUG_FONT_SIZE = 16


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.FreeTypeFont:
    """폰트는 크기별로 한 번만 불러온다."""
    return ImageFont.truetype(FONT_PATH, size)


@lru_cache(maxsize=256)
def _fit_size(text: str, max_size: int, max_width: int) -> int:
    """max_size에서 시작해 글자 폭이 max_width 안에 들어올 때까지 줄인다."""
    size = max_size
    while size > 10:
        left, _, right, _ = _font(size).getbbox(text)
        if right - left <= max_width:
            break
        size -= 1
    return size


def show_message(frame: np.ndarray, text: str, kind: str) -> np.ndarray:
    """프레임 아래쪽에 안내 배너를 그린 새 프레임을 돌려준다. kind: welcome / counter / hint"""
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = int(h * BANNER_RATIO)
    top = h - banner_h

    # 반투명 배너
    color = np.array(BANNER_COLORS.get(kind, BANNER_COLORS["hint"]), dtype=np.float32)
    region = out[top:].astype(np.float32)
    out[top:] = (region * (1 - BANNER_ALPHA) + color * BANNER_ALPHA).astype(np.uint8)

    # 한글 글자는 Pillow로 배너 부분에만 그림 (BGR → RGB → PIL → BGR)
    size = _fit_size(text, max(int(banner_h * TEXT_RATIO), 11), w - 2 * TEXT_MARGIN)
    font = _font(size)
    pil = Image.fromarray(cv2.cvtColor(out[top:], cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    left, upper, right, lower = draw.textbbox((0, 0), text, font=font)
    x = (w - (right - left)) // 2 - left
    y = (banner_h - (lower - upper)) // 2 - upper
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    out[top:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return out


def setup_window(fullscreen: bool = True) -> None:
    """안내 화면 창을 만든다. 기본은 전체 화면."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    set_fullscreen(fullscreen)


def set_fullscreen(on: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def toggle_fullscreen() -> bool:
    """전체 화면을 켜고 끈다. 바뀐 뒤 전체 화면이면 True."""
    now_full = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
    set_fullscreen(not now_full)
    return not now_full


def mirror(frame: np.ndarray) -> np.ndarray:
    """화면을 거울처럼 좌우로 뒤집은 새 프레임."""
    return cv2.flip(frame, 1)


def mirror_bbox(bbox: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    """원본 좌표의 박스를 좌우로 뒤집힌 화면의 좌표로 바꾼다."""
    x1, y1, x2, y2 = bbox
    return (width - 1 - x2, y1, width - 1 - x1, y2)


def draw_debug(frame: np.ndarray, people: list[PersonBox],
               results: dict[int, RecognitionResult] | None = None,
               animals: list | None = None, fps: float | None = None,
               mirrored: bool = False) -> np.ndarray:
    """사람 박스, 추적 번호, A의 인식 결과(이름·거리)를 그린 새 프레임을 돌려준다.

    results: 추적 번호별 최근 인식 결과. animals: detector.last_animals
    mirrored: True면 원본 frame을 좌우로 뒤집고, 박스는 뒤집힌 위치에 그린다 (글자는 정상 방향).
    """
    out = mirror(frame) if mirrored else frame.copy()
    w = out.shape[1]
    labels = []   # (x, y, 글자, BGR 색)

    for p in people:
        x1, y1, x2, y2 = mirror_bbox(p.bbox, w) if mirrored else p.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), DEBUG_PERSON, 2)
        text = f"ID {p.track_id}"
        r = (results or {}).get(p.track_id)
        if r is not None:
            if not r.face_found:
                text += " 얼굴 없음"
            else:
                text += f" {r.name or '미등록'}"
                if r.distance is not None:
                    text += f" {r.distance:.2f}"
        labels.append((x1, y1, text, DEBUG_PERSON))

    for name, _, bbox, _ in animals or []:
        x1, y1, x2, y2 = mirror_bbox(bbox, w) if mirrored else bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), DEBUG_ANIMAL, 2)
        labels.append((x1, y1, f"{name} (무시)", DEBUG_ANIMAL))

    if fps is not None:   # 오른쪽 위 (x=None)
        labels.append((None, 4 + DEBUG_FONT_SIZE + 6, f"FPS {fps:.1f}", (0, 255, 255)))

    if labels:
        font = _font(DEBUG_FONT_SIZE)
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for x, y, text, (b, g, r_) in labels:
            if x is None:
                l, _, rr, _ = draw.textbbox((0, 0), text, font=font)
                x = out.shape[1] - (rr - l) - 8
            ty = max(y - DEBUG_FONT_SIZE - 6, 0)
            l, t, rr, bb = draw.textbbox((x, ty), text, font=font)
            draw.rectangle((l - 2, t - 2, rr + 2, bb + 2), fill=(0, 0, 0))
            draw.text((x, ty), text, font=font, fill=(r_, g, b))
        out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return out
