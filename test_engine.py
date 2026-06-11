"""1단계 엔진 검증 — 합성 장면 + EXIF 회전.

검증 목표:
  A) detect_quad: 합성 장면에 심은 사각형의 네 꼭짓점을 ~오차 내로 찾는다.
  B) warp:        보정 결과가 원래 문서 비율에 근접한 직사각형이다.
  C) load_image:  EXIF Orientation=6(90°) 사진을 똑바로 세워 읽는다.
실행: python test_engine.py        (선택) python test_engine.py <실사진경로>
"""

import io
import sys
import tempfile
import os

import numpy as np
import cv2
from PIL import Image

from scanner.engine import load_image, detect_quad, warp


def make_document(w=400, h=560):
    """흰 바탕 + 검은 글자/선이 있는 가짜 문서."""
    doc = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(doc, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)
    for i, y in enumerate(range(40, h - 40, 28)):
        x2 = w - 40 if i % 3 else w - 120
        cv2.line(doc, (40, y), (x2, y), (30, 30, 30), 3)
    return doc


def make_scene():
    """문서를 원근 변형해 배경에 심고, 심은 네 꼭짓점(ground truth)을 함께 반환."""
    bg = np.full((900, 1200, 3), 160, np.uint8)
    cv2.randu(bg, 140, 180)  # 약한 질감
    doc = make_document()
    h, w = doc.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    # 배경 위 목표 사각형(기울어진 원근, 세로형 — 문서 400x560 비율과 맞춤)
    dst = np.array([[330, 130], [760, 190], [700, 810], [280, 760]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, m, (bg.shape[1], bg.shape[0]))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m,
                               (bg.shape[1], bg.shape[0]))
    bg[mask > 0] = warped[mask > 0]
    return bg, dst


def corner_error(detected, truth):
    """정렬 후 꼭짓점 평균 거리(px)."""
    from scanner.engine import order_points
    d = order_points(detected)
    t = order_points(truth)
    return float(np.mean(np.linalg.norm(d - t, axis=1)))


def test_detect_and_warp():
    scene, truth = make_scene()
    quad = detect_quad(scene)
    assert quad is not None, "detect_quad 가 사각형을 못 찾음"
    err = corner_error(quad, truth)
    print(f"[A] 꼭짓점 평균 오차: {err:.1f}px (기준: < 15)")
    assert err < 15, f"꼭짓점 오차 과대: {err:.1f}px"

    out = warp(scene, quad)
    oh, ow = out.shape[:2]
    ratio = oh / ow
    print(f"[B] 보정 결과 {ow}x{oh}, 세로/가로 비율={ratio:.2f} (문서 원본 560/400=1.40)")
    assert 1.2 < ratio < 1.6, f"보정 비율 이상: {ratio:.2f}"
    print("[A][B] 통과")


def test_exif_orientation():
    # 가로 600x400 빨강/파랑 좌우 분할 이미지에 Orientation=6(시계 90°) 태그
    arr = np.zeros((400, 600, 3), np.uint8)
    arr[:, :300] = (0, 0, 255)     # 왼쪽 빨강(RGB)
    arr[:, 300:] = (255, 0, 0)     # 오른쪽 파랑(RGB)
    pil = Image.fromarray(arr)
    exif = pil.getexif()
    exif[274] = 6  # Orientation
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    pil.save(path, exif=exif)
    try:
        img = load_image(path)  # BGR
        h, w = img.shape[:2]
        print(f"[C] EXIF=6 로드 결과 {w}x{h} (기대: 세로가 더 긴 400x600)")
        assert h > w, "EXIF 회전이 반영되지 않음(가로 그대로)"
        print("[C] 통과 — EXIF 회전 자동 반영 확인")
    finally:
        os.remove(path)


def test_real_photo(path):
    img = load_image(path)
    print(f"\n[실사진] {path}  {img.shape[1]}x{img.shape[0]}")
    quad = detect_quad(img)
    if quad is None:
        print("  자동 인식 실패 — 수동 보정 필요(예상된 동작일 수 있음)")
        return
    print(f"  인식 꼭짓점:\n{quad}")
    out = warp(img, quad)
    outp = os.path.splitext(path)[0] + "_scanned.jpg"
    from scanner.engine import save_image
    save_image(outp, out)
    print(f"  보정 저장: {outp}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    test_detect_and_warp()
    test_exif_orientation()
    if len(sys.argv) > 1:
        test_real_photo(sys.argv[1])
    print("\n전체 통과")
