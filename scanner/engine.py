"""변환 엔진 — UI 없는 순수 함수 모음.

핵심 3종:
  - load_image(path)        : EXIF 회전을 반영해 BGR 이미지로 읽는다.
  - detect_quad(image)      : 사각형 네 꼭짓점을 추정한다(실패 시 None).
  - warp(image, quad)       : 네 점을 기준으로 투시 보정한 이미지를 만든다.

자동 인식은 classic CV(엣지→윤곽선→4점 근사)로 'best guess'만 낸다.
LCD/모니터처럼 발광·반사로 윤곽선이 끊기는 대상은 실패율이 높으므로,
수동 보정(GUI 단계)이 메인 경로라는 전제로 설계한다.
"""

from __future__ import annotations

import numpy as np
import cv2


# ── 입력 ────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """경로에서 BGR 이미지를 읽는다.

    cv2.imread 는 기본적으로 EXIF Orientation 태그를 반영해 자동 회전한다
    (회전을 끄려면 IMREAD_IGNORE_ORIENTATION 플래그를 줘야 함). 따라서 폰 사진의
    가로/세로가 눕는 문제는 기본 동작으로 처리된다.

    한글·공백 경로에서 cv2.imread 가 실패하는 경우가 있어 np.fromfile 로 우회한다.
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # EXIF 회전 반영
    if img is None:
        raise ValueError(f"이미지를 디코드할 수 없습니다: {path}")
    return img


def save_image(path: str, image: np.ndarray, quality: int = 95) -> None:
    """한글·공백 경로 대응 저장(JPG/PNG). 확장자로 포맷 결정."""
    ext = "." + path.rsplit(".", 1)[-1].lower()
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"이미지를 인코드할 수 없습니다: {path}")
    buf.tofile(path)


# ── 자동 인식 ────────────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """4점을 (좌상, 우상, 우하, 좌하) 순서로 정렬."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 좌상: x+y 최소
    rect[2] = pts[np.argmax(s)]   # 우하: x+y 최대
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # 우상: x-y 최소
    rect[3] = pts[np.argmax(diff)]  # 좌하: x-y 최대
    return rect


def detect_quad(image: np.ndarray, *, max_dim: int = 1000) -> np.ndarray | None:
    """가장 그럴듯한 사각형 네 꼭짓점을 원본 좌표계로 반환. 실패 시 None.

    절차: 축소 → 그레이/블러 → Canny → 팽창(끊긴 엣지 연결) → 윤곽선 →
          면적 큰 순으로 4점 볼록 다각형 탐색.
    """
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = small.shape[0] * small.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    best: np.ndarray | None = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.10:        # 화면의 10% 미만은 무시
            continue
        if area > img_area * 0.95:        # 사진 테두리 전체(프레임) 오검출 방지
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx.reshape(4, 2).astype(np.float32)
            break

    if best is None:
        return None
    return order_points(best / scale)   # 원본 좌표계로 복원


# ── 투시 보정 ────────────────────────────────────────────────────────────────

def warp(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """네 점(quad)을 직사각형으로 펴는 투시 보정 결과를 반환."""
    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = int(round(max(width_top, width_bottom)))
    out_h = int(round(max(height_left, height_right)))
    out_w = max(out_w, 1)
    out_h = max(out_h, 1)

    dst = np.array([[0, 0],
                    [out_w - 1, 0],
                    [out_w - 1, out_h - 1],
                    [0, out_h - 1]], dtype=np.float32)

    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (out_w, out_h))


# ── 후처리(보정 품질) ───────────────────────────────────────────────────────

def enhance(image: np.ndarray, *, sharpen: bool = False,
            auto_contrast: bool = False, grayscale: bool = False) -> np.ndarray:
    """선택적 후처리. 순서: 그레이스케일 → 명암 보정 → 샤프닝."""
    out = image
    if grayscale:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)  # 저장/미리보기 일관성 위해 3채널 유지
    if auto_contrast:
        # LAB의 L(밝기) 채널을 1~99 퍼센타일로 스트레치(색조 유지)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lo, hi = np.percentile(l, (1, 99))
        if hi > lo + 1:
            l = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo),
                        0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if sharpen:
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)  # 언샤프 마스크
    return out
