"""Photo Scanner — 통합 GUI (단일·일괄 통합).

흐름: [사진 열기](1장/여러 장) 또는 [폴더 열기] → 전체 자동 인식 → 왼쪽 썸네일
      목록에서 한 장씩 → 빈 곳 클릭(가장 가까운 꼭짓점 이동) / 점 드래그(미세조정)
      → 오른쪽 실시간 미리보기로 확인 → Enter(이 장 확인+다음) → [전체 내보내기]
      → 각 원본 옆 _scanned/ 에 JPG+PNG.

좌표계: QGraphicsScene 을 이미지 픽셀 좌표(1:1)로 쓴다. 핸들의 scene 좌표가 곧
이미지 좌표라 engine.warp 에 그대로 넘긴다.

메모리: 고해상도 원본을 전부 들고 있지 않는다. 사전 패스에서 한 장씩 열어 꼭짓점만
추정하고(썸네일만 보관) 해제. 화면/내보내기 때만 다시 디코드한다.

실행:  python -m scanner.app  [파일 또는 폴더]
자체검증: python -m scanner.app --selftest <폴더>
"""

from __future__ import annotations

import html
import os
import sys

import numpy as np
import cv2
from PIL import Image as PILImage
from PySide6 import QtCore, QtGui, QtWidgets

from scanner.engine import load_image, detect_quad, warp, save_image, enhance

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_EXIF_IMAGE_DESCRIPTION = 0x010E  # 아이폰 사진 "설명/캡션"이 들어오는 EXIF 태그


def read_label(path: str) -> str | None:
    """EXIF ImageDescription(아이폰 캡션)을 읽어 라벨 문자열로 돌려준다(없으면 None).

    아이폰은 이 값을 UTF-8 바이트로 넣는데, Pillow 는 이를 latin-1 로 해석해
    str 로 돌려준다(→ 한글이 깨진 모양). 그 경우 latin-1 로 되돌려 UTF-8 로 다시
    디코드해 원문을 복원한다. 실패하면 조용히 None."""
    try:
        v = PILImage.open(path).getexif().get(_EXIF_IMAGE_DESCRIPTION)
    except Exception:
        return None
    if not v:
        return None
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8").strip() or None
        except Exception:
            return None
    if isinstance(v, str):
        try:
            return (v.encode("latin-1").decode("utf-8").strip() or None)
        except Exception:
            return v.strip() or None
    return None


_ILLEGAL_FS = '<>:"/\\|?*'  # 윈도우 파일명 금지 문자


def safe_stem(name: str) -> str:
    """라벨을 파일명으로 쓸 수 있게 정리(금지문자 제거, 끝의 점/공백 정리)."""
    out = "".join("_" if c in _ILLEGAL_FS else c for c in name)
    return out.strip().rstrip(". ").strip() or "untitled"

# 꼭짓점 색/번호 (좌상=1 … 좌하=4)
HANDLE_COLORS = [
    QtGui.QColor(230, 50, 50),    # 1 좌상 빨강
    QtGui.QColor(40, 200, 90),    # 2 우상 초록
    QtGui.QColor(60, 130, 255),   # 3 우하 파랑
    QtGui.QColor(255, 175, 0),    # 4 좌하 주황
]
PREVIEW_MAX = 900  # 실시간 미리보기/연산용 축소 한도(px)


def bgr_to_qpixmap(bgr: np.ndarray) -> QtGui.QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _fmt_size(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _folder_images(d: str) -> list[str]:
    """폴더 안 이미지 파일 경로를 정렬해 반환."""
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(d, f)))


def default_quad(w: int, h: int) -> np.ndarray:
    mx, my = w * 0.1, h * 0.1
    return np.array([(mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my)],
                    dtype=np.float32)


def make_thumb(bgr: np.ndarray, s: int = 64) -> QtGui.QIcon:
    h, w = bgr.shape[:2]
    sc = s / float(max(h, w))
    small = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                       interpolation=cv2.INTER_AREA)
    return QtGui.QIcon(bgr_to_qpixmap(small))


def emoji_icon(ch: str, px: int = 20) -> QtGui.QIcon:
    """이모지 글리프를 아이콘으로 렌더(목록의 🏷 표식과 시각적 통일용)."""
    pm = QtGui.QPixmap(px, px)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    f = p.font()
    f.setPointSizeF(px * 0.68)
    p.setFont(f)
    p.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, ch)
    p.end()
    return QtGui.QIcon(pm)


class CornerHandle(QtWidgets.QGraphicsItem):
    """ㄱ자(코너 브래킷) 핸들 — 꼭짓점 지점은 비워 두고 두 팔이 사각형 안쪽으로
    뻗는다. 가는 십자선이 정확한 꼭짓점 픽셀을 가리킨다. 색으로 구분(좌상 빨강,
    우상 초록, 우하 파랑, 좌하 주황). scene 경계로 클램프."""

    ARM = 24     # 팔 길이(화면 px — ItemIgnoresTransformations)
    GAP = 5      # 꼭짓점 주변 비우는 간격
    THICK = 4    # 팔 두께
    # idx별 팔 방향(사각형 안쪽): 좌상(+,+) 우상(-,+) 우하(-,-) 좌하(+,-)
    DIRS = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    def __init__(self, idx: int, radius_unused: float, on_move):
        super().__init__()
        self.idx = idx
        self._on_move = on_move
        self.setZValue(10)
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def boundingRect(self) -> QtCore.QRectF:
        m = self.ARM + 3
        return QtCore.QRectF(-m, -m, 2 * m, 2 * m)  # 잡기 영역 넉넉히

    def paint(self, painter, option, widget=None):
        col = HANDLE_COLORS[self.idx]
        dx, dy = self.DIRS[self.idx]
        # 십자선(가늘게) — 정확한 꼭짓점 표시, 중심 픽셀은 비움
        thin = QtGui.QPen(QtGui.QColor(col.red(), col.green(), col.blue(), 190), 1)
        painter.setPen(thin)
        c = self.GAP - 2
        painter.drawLine(QtCore.QPointF(-self.ARM * 0.5, 0), QtCore.QPointF(-c, 0))
        painter.drawLine(QtCore.QPointF(c, 0), QtCore.QPointF(self.ARM * 0.5, 0))
        painter.drawLine(QtCore.QPointF(0, -self.ARM * 0.5), QtCore.QPointF(0, -c))
        painter.drawLine(QtCore.QPointF(0, c), QtCore.QPointF(0, self.ARM * 0.5))
        # ㄱ자 브래킷(두껍게) — 흰 외곽선 위에 색 본체(어두운/밝은 배경 모두 가시)
        for pen in (QtGui.QPen(QtGui.QColor(255, 255, 255), self.THICK + 2),
                    QtGui.QPen(col, self.THICK)):
            pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(self.GAP * dx, 0),
                             QtCore.QPointF(self.ARM * dx, 0))
            painter.drawLine(QtCore.QPointF(0, self.GAP * dy),
                             QtCore.QPointF(0, self.ARM * dy))
        # 번호 동그라미 — L자 안쪽에 배치(꼭짓점은 비워 정밀 클릭 유지)
        r = 8.0
        cx, cy = dx * 15.0, dy * 15.0
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.5))
        painter.setBrush(QtGui.QBrush(col))
        painter.drawEllipse(QtCore.QPointF(cx, cy), r, r)
        f = painter.font()
        f.setBold(True)
        f.setPointSizeF(r)
        painter.setFont(f)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.drawText(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r),
                         QtCore.Qt.AlignmentFlag.AlignCenter, str(self.idx + 1))

    def shape(self) -> QtGui.QPainterPath:
        """잡기(드래그) 영역 = 두 팔 + 번호 동그라미. 꼭짓점 *중심*은 여전히
        비워(원은 중심에서 15px 안쪽), 그 지점 클릭은 '가장 가까운 꼭짓점 이동'이
        동작한다(정밀 클릭 유지). 동그라미는 가장 눈에 띄는 타깃이라 같이 잡게 함."""
        path = QtGui.QPainterPath()
        dx, dy = self.DIRS[self.idx]
        pad = 8.0
        x0, x1 = sorted((self.GAP * dx, self.ARM * dx))
        path.addRect(QtCore.QRectF(x0, -pad, x1 - x0, 2 * pad))      # 가로 팔
        y0, y1 = sorted((self.GAP * dy, self.ARM * dy))
        path.addRect(QtCore.QRectF(-pad, y0, 2 * pad, y1 - y0))      # 세로 팔
        path.addEllipse(QtCore.QPointF(dx * 15.0, dy * 15.0), 10.0, 10.0)  # 번호 원
        return path

    def itemChange(self, change, value):
        Flag = QtWidgets.QGraphicsItem.GraphicsItemChange
        if change == Flag.ItemPositionChange and self.scene() is not None:
            r = self.scene().sceneRect()
            value.setX(min(max(value.x(), r.left()), r.right()))
            value.setY(min(max(value.y(), r.top()), r.bottom()))
            return value
        if change == Flag.ItemPositionHasChanged:
            self._on_move(self.idx)
        return super().itemChange(change, value)


class ScanView(QtWidgets.QGraphicsView):
    """캔버스: 이미지 + 4 모서리(드래그/클릭) + 줌/팬 + 돋보기 루페."""

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QtGui.QColor(34, 34, 34))  # 다크 캔버스 배경

        self.bgr: np.ndarray | None = None
        self.handles: list[CornerHandle] = []
        self._pixmap = QtGui.QPixmap()
        self._poly = QtWidgets.QGraphicsPolygonItem()
        self._small: np.ndarray | None = None      # 미리보기용 축소본
        self._small_scale = 1.0
        self._user_zoomed = False
        self._min_scale = 1.0
        self._panning = False
        self._pan_last = QtCore.QPoint()
        self._active = 0
        self._mouse_scene = QtCore.QPointF()       # 돋보기 중심(마우스 위치 추종)

        self.on_quad_changed = lambda: None        # 윈도우가 주입(미리보기 갱신)
        self.on_prev = lambda: None                # 좌우 화살표 → 윈도우가 주입
        self.on_next = lambda: None
        self.loupe_enabled = True
        self._loupe = QtWidgets.QLabel(self.viewport())
        self._loupe.setFixedSize(180, 180)
        self._loupe.setStyleSheet(
            "border:2px solid #222; background:#282828;")
        self._loupe.hide()
        # 캔버스 좌우 가장자리 이전/다음 화살표(이미지 위 오버레이)
        # on_prev/on_next 는 윈도우가 나중에 주입하므로 람다로 늦게 호출
        self._nav_prev = self._make_nav("‹", lambda: self.on_prev())
        self._nav_next = self._make_nav("›", lambda: self.on_next())

    def _make_nav(self, text: str, slot) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton(self.viewport())
        b.setText(text)
        b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        b.setFixedSize(38, 64)
        b.setStyleSheet(
            "QToolButton{background:rgba(20,20,20,140); color:#eee; border:none;"
            " border-radius:8px; font-size:26px; font-weight:bold;}"
            "QToolButton:hover{background:rgba(45,127,249,210);}")
        b.clicked.connect(slot)
        b.hide()
        return b

    def _place_nav(self):
        vw, vh = self.viewport().width(), self.viewport().height()
        y = vh // 2 - 32
        self._nav_prev.move(8, y)
        self._nav_next.move(vw - 46, y)
        show = self.bgr is not None
        self._nav_prev.setVisible(show)
        self._nav_next.setVisible(show)
        self._nav_prev.raise_()
        self._nav_next.raise_()

    # ── 표시 ──
    def set_image(self, bgr: np.ndarray):
        self.bgr = bgr
        self._user_zoomed = False
        sc = self.scene()
        sc.clear()
        self.handles = []
        h, w = bgr.shape[:2]
        self._pixmap = bgr_to_qpixmap(bgr)
        sc.addPixmap(self._pixmap)
        sc.setSceneRect(0, 0, w, h)
        # 미리보기용 축소본
        s = min(1.0, PREVIEW_MAX / float(max(h, w)))
        self._small = (cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                       if s < 1.0 else bgr.copy())
        self._small_scale = s
        # 폴리곤
        self._poly = QtWidgets.QGraphicsPolygonItem()
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
        pen.setCosmetic(True)
        pen.setWidth(1)
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        self._poly.setPen(pen)
        self._poly.setZValue(5)
        sc.addItem(self._poly)
        self._make_handles([(x, y) for x, y in default_quad(w, h)])
        self.fit()
        self._place_nav()

    def _make_handles(self, pts):
        for hd in self.handles:
            self.scene().removeItem(hd)
        self.handles = []
        for i, (x, y) in enumerate(pts):
            hd = CornerHandle(i, 11.0, self._handle_moved)
            hd.setPos(float(x), float(y))
            self.scene().addItem(hd)
            self.handles.append(hd)
        self._update_poly()
        self.on_quad_changed()

    def _handle_moved(self, idx: int):
        if len(self.handles) != 4:   # 핸들 구성 중간 콜백 무시
            return
        self._active = idx
        self._update_poly()
        self.on_quad_changed()

    def _update_poly(self):
        if len(self.handles) == 4:
            self._poly.setPolygon(QtGui.QPolygonF([h.pos() for h in self.handles]))

    def set_quad(self, quad: np.ndarray):
        self._make_handles([(float(x), float(y)) for x, y in quad])

    def quad(self) -> np.ndarray:
        return np.array([[h.pos().x(), h.pos().y()] for h in self.handles],
                        dtype=np.float32)

    def preview_image(self) -> np.ndarray | None:
        """축소본 기준으로 펴진 결과(빠름). 미리보기용."""
        if self._small is None or len(self.handles) != 4:
            return None
        return warp(self._small, self.quad() * self._small_scale)

    # ── 클릭 = 가장 가까운 꼭짓점 이동 ──
    def _nearest(self, scene_pt: QtCore.QPointF) -> int:
        p = np.array([scene_pt.x(), scene_pt.y()])
        d = [np.hypot(*(np.array([h.pos().x(), h.pos().y()]) - p)) for h in self.handles]
        return int(np.argmin(d))

    def place_nearest(self, scene_pt: QtCore.QPointF):
        if not self.handles:
            return
        idx = self._nearest(scene_pt)
        self._active = idx
        self.handles[idx].setPos(scene_pt)  # itemChange → _handle_moved

    # ── 줌/팬 ──
    def fit(self):
        if self.scene().sceneRect().isValid():
            self.fitInView(self.scene().sceneRect(),
                           QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self._min_scale = self.transform().m11()

    def reset_fit(self):
        self._user_zoomed = False
        self.fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._user_zoomed:
            self.fit()
        self._place_nav()

    def _zoom(self, factor: float) -> bool:
        if self.bgr is None:
            return False
        nxt = self.transform().m11() * factor
        if nxt < self._min_scale * 0.999 or nxt > 8.0:
            return False
        self._user_zoomed = True
        self.scale(factor, factor)
        return True

    def wheelEvent(self, e):
        self._zoom(1.25 if e.angleDelta().y() > 0 else 1 / 1.25)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last = e.position().toPoint()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            e.accept()
            return
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self.bgr is not None:
            self._mouse_scene = self.mapToScene(e.position().toPoint())
            it = self.itemAt(e.position().toPoint())
            if not isinstance(it, CornerHandle):  # 빈 곳/이미지/선 클릭 → 가까운 꼭짓점
                self.place_nearest(self.mapToScene(e.position().toPoint()))
                self._update_loupe()
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._panning:
            p = e.position().toPoint()
            d = p - self._pan_last
            self._pan_last = p
            hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
            hb.setValue(hb.value() - d.x())
            vb.setValue(vb.value() - d.y())
            e.accept()
            return
        if self.bgr is not None and self.handles:
            self._mouse_scene = self.mapToScene(e.position().toPoint())
            self._active = self._nearest(self._mouse_scene)
            self._update_loupe()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._panning and e.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        self.reset_fit()
        e.accept()

    def leaveEvent(self, e):
        self._loupe.hide()
        super().leaveEvent(e)

    # ── 돋보기 루페 ──
    def set_loupe_enabled(self, on: bool):
        self.loupe_enabled = on
        if not on:
            self._loupe.hide()

    def _update_loupe(self):
        if not self.loupe_enabled or self.bgr is None or not self.handles:
            return
        crop = 70  # 중심 기준 ±70px 영역
        c = self._mouse_scene  # 마우스 위치를 확대(꼭짓점이 곧 마우스인 드래그 중에도 일치)
        tile = QtGui.QPixmap(2 * crop, 2 * crop)
        tile.fill(QtGui.QColor(40, 40, 40))
        p = QtGui.QPainter(tile)
        p.drawPixmap(0, 0, self._pixmap,
                     int(c.x() - crop), int(c.y() - crop), 2 * crop, 2 * crop)
        col = HANDLE_COLORS[self._active]
        p.setPen(QtGui.QPen(col, 2))
        p.drawLine(crop - 12, crop, crop + 12, crop)
        p.drawLine(crop, crop - 12, crop, crop + 12)
        p.end()
        size = self._loupe.width()
        self._loupe.setPixmap(tile.scaled(
            size, size, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        # 활성 꼭짓점 반대쪽 위 모서리에 배치(작업영역 안 가림)
        vw = self.viewport().width()
        hv = self.mapFromScene(c)
        x = 10 if hv.x() > vw * 0.6 else vw - size - 10
        self._loupe.move(x, 10)
        self._loupe.show()


class PreviewLabel(QtWidgets.QLabel):
    """원본 픽스맵을 보관하고, 라벨 크기가 바뀔 때마다 거기에 맞춰 다시 스케일.

    (픽스맵이 라벨 최소 크기를 역으로 키우는 피드백 루프를 막기 위해
    sizePolicy 를 Ignored 로 두고, 스케일은 항상 원본에서 다시 계산한다.)
    """

    def __init__(self, text: str = ""):
        super().__init__(text)
        self._src: QtGui.QPixmap | None = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Ignored)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def set_source(self, pm: QtGui.QPixmap):
        self._src = pm
        self._rescale()

    def _rescale(self):
        if self._src is None or self.width() < 16 or self.height() < 16:
            return
        super().setPixmap(self._src.scaled(
            self.width() - 8, self.height() - 8,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()


class HtmlDelegate(QtWidgets.QStyledItemDelegate):
    """목록 항목 텍스트를 HTML 로 그린다 — 한 항목 안에서 줄마다 다른 글꼴/색
    (파일명=굵은 흰색, 태그·배지=옅은 색·작게)을 쓰기 위함."""

    def _doc(self, htmltext: str, font: QtGui.QFont) -> QtGui.QTextDocument:
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(font)
        doc.setHtml(htmltext)
        return doc

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        text = opt.text                        # 비우기 전에 HTML 보관
        opt.text = ""                          # 텍스트는 우리가 직접(HTML) 그림
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
                          opt, painter, opt.widget)
        rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        doc = self._doc(text, opt.font)
        doc.setTextWidth(rect.width())
        painter.save()
        painter.translate(rect.topLeft())
        doc.drawContents(painter, QtCore.QRectF(0, 0, rect.width(), rect.height()))
        painter.restore()

    def sizeHint(self, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        w = opt.rect.width() if opt.rect.width() > 0 else 200
        doc = self._doc(opt.text, opt.font)
        doc.setTextWidth(w - opt.decorationSize.width() - 12)
        h = max(doc.size().height() + 6, opt.decorationSize.height() + 8)
        return QtCore.QSize(w, int(h))


class Page:
    __slots__ = ("path", "quad", "confirmed", "thumb", "label",
                 "sharpen", "contrast", "gray")

    def __init__(self, path: str):
        self.path = path
        self.quad: np.ndarray | None = None
        self.confirmed = False
        self.thumb: QtGui.QIcon | None = None
        self.label: str | None = None   # EXIF 라벨(아이폰 캡션) — 일괄 이름변경 원본
        self.sharpen = False            # 사진별 보정 옵션(개별)
        self.contrast = False
        self.gray = False


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Scanner")
        self.resize(1400, 880)
        self.pages: list[Page] = []
        self.idx = -1
        self._loading = False

        # 좌: 썸네일 목록
        self.listw = QtWidgets.QListWidget()
        self.listw.setMaximumWidth(220)
        self.listw.setIconSize(QtCore.QSize(72, 72))
        # Shift/Ctrl 다중선택 + Ctrl+A 전체선택(라벨 일괄 이름변경 대상 고르기용)
        self.listw.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.listw.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.listw.customContextMenuRequested.connect(self._list_menu)
        self.listw.currentRowChanged.connect(self.goto)
        self.listw.setItemDelegate(HtmlDelegate(self.listw))  # 줄별 글꼴 구분
        # 목록 위: 표식 필터 + 전체선택(보이는 것만)
        left_box = QtWidgets.QWidget()
        left_box.setMaximumWidth(220)
        lv = QtWidgets.QVBoxLayout(left_box)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(3)
        left_box.setStyleSheet(self.LEFT_QSS)
        # 표식 필터 — 토글 버튼 OR(여러 표식 동시 표시). 기본 모두 켬=전체.
        filt_grp = QtWidgets.QGroupBox("표식 필터")
        fl = QtWidgets.QHBoxLayout(filt_grp)
        fl.setContentsMargins(8, 4, 8, 8)
        fl.setSpacing(4)
        self.cb_f_fail = self._make_filter_btn("? 미인식", "자동인식 실패만 모아 빠르게 처리")
        self.cb_f_det = self._make_filter_btn("• 인식", "인식됨(미확인) — 결과 빠른 검토")
        self.cb_f_conf = self._make_filter_btn("✓ 확인", "확인 완료 — 2차 검증")
        for b in (self.cb_f_fail, self.cb_f_det, self.cb_f_conf):
            fl.addWidget(b)
        # 전체선택/해제 토글
        self.btn_all = QtWidgets.QPushButton("전체 선택")
        self.btn_all.setCheckable(True)
        self.btn_all.setToolTip("현재 필터로 보이는 사진 전체 선택 / 다시 누르면 해제")
        self.btn_all.toggled.connect(self._toggle_select_all)
        lv.addWidget(filt_grp)
        lv.addWidget(self.btn_all)
        lv.addWidget(self.listw, 1)
        # 중앙: 캔버스
        self.view = ScanView()
        self.view.on_quad_changed = self.update_preview
        self.view.setAcceptDrops(False)  # 드롭은 메인 윈도우가 받는다(아래 dropEvent)
        # 우: 실시간 미리보기
        prev_box = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(prev_box)
        pv.setContentsMargins(6, 6, 6, 6)
        pv.addWidget(QtWidgets.QLabel("실시간 미리보기"))
        self.preview = PreviewLabel("(사진을 여세요)")
        self.preview.setMinimumWidth(280)
        self.preview.setStyleSheet("background:#1e1e1e; color:#aaa;")
        pv.addWidget(self.preview, 1)
        pv.addWidget(self._build_options())

        # 캔버스 좌우 가장자리 이전/다음 화살표(상단바 버튼과 겸용)
        self.view.on_prev = self.prev
        self.view.on_next = self.next

        split = QtWidgets.QSplitter()
        split.addWidget(left_box)
        split.addWidget(self.view)
        split.addWidget(prev_box)
        split.setStretchFactor(1, 3)  # 창 확대 시 캔버스:미리보기 = 3:1 로 분배
        split.setStretchFactor(2, 1)
        split.setSizes([200, 880, 320])

        # 하단 조작 안내바
        self.helpbar = QtWidgets.QLabel(
            "  F: 화면에 맞춤   ·   Enter: 수정완료+다음   ·   ← →: 이전/다음   ·   "
            "F2: 이름변경   ·   Del: 목록제거   ·   1·2·3: 샤프닝·명암·흑백")
        self.helpbar.setStyleSheet("background:#252525; color:#9a9a9a; padding:4px;")

        container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(split, 1)
        lay.addWidget(self.helpbar)
        self.setCentralWidget(container)

        self.status = self.statusBar()
        self._build_toolbar()
        self._build_shortcuts()
        self.setAcceptDrops(True)  # 파일/폴더 드래그앤드랍

    # 버튼 종류별 어포던스: 일회성=평평한 버튼+아이콘 / 토글=체크 시 파란 배경+문구 변화
    # / 주요 완료 액션=강조색. 기능군은 구분선으로 묶는다.
    TOOLBAR_QSS = """
        QToolBar { spacing: 6px; padding: 5px; background: #2f2f2f;
                   border-bottom: 1px solid #1c1c1c; }
        QToolBar::separator { width: 1px; background: #4a4a4a; margin: 4px 6px; }
        QToolButton { padding: 5px 10px; border-radius: 6px;
                      border: 1px solid transparent; color: #dddddd; }
        QToolButton:hover { background: #3f3f3f; }
        QToolButton:pressed { background: #4a4a4a; }
        QToolButton:checked { background: #2d7ff9; color: white;
                              border: 1px solid #1b62c9; }
        QToolButton#primary { background: #1f9d55; color: white; font-weight: bold;
                              border: 1px solid #157040; }
        QToolButton#primary:hover { background: #23b261; }
        QToolButton#primary:pressed { background: #188047; }
    """

    # 좌측 패널 버튼(필터 토글 · 전체선택). 켜짐=파란 채움(툴바 토글과 동일 어포던스).
    LEFT_QSS = """
        QPushButton { padding: 6px 4px; border-radius: 6px; font-size: 11px;
                      border: 1px solid #4a4a4a; background: #3a3a3a; color: #cccccc; }
        QPushButton:hover { background: #454545; }
        QPushButton:checked { background: #2d7ff9; color: white; font-weight: bold;
                              border: 1px solid #1b62c9; }
        QPushButton:checked:hover { background: #3b8bff; }
    """

    def _build_toolbar(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.setStyleSheet(self.TOOLBAR_QSS)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        std = self.style().standardIcon
        SP = QtWidgets.QStyle.StandardPixmap

        def act(icon, text, slot, tip=""):
            qicon = icon if isinstance(icon, QtGui.QIcon) else std(icon)
            a = QtGui.QAction(qicon, text, self)
            a.triggered.connect(slot)
            if tip:
                a.setToolTip(tip)
            tb.addAction(a)
            return a

        # 1) 열기 (일회성: 대화상자 팝업)
        act(emoji_icon("🖼️"), "사진 열기", self.open_files, "사진 파일 선택(여러 장 가능)")
        act(emoji_icon("📁"), "폴더 열기", self.open_folder, "폴더 안 이미지 전부 열기")
        tb.addSeparator()
        # 2) 탐색 (일회성: 즉시 동작)
        act(SP.SP_ArrowLeft, "이전", self.prev, "이전 사진 (←)")
        act(SP.SP_ArrowRight, "다음", self.next, "다음 사진 (→)")
        tb.addSeparator()
        # 2.5) 이름 (일회성: 즉시 파일명 변경)
        act(emoji_icon("🏷"), "라벨로 이름변경", self.rename_to_label,
            "선택한 사진들의 파일명을 EXIF 라벨로 일괄 변경 (Shift/Ctrl 다중선택, Ctrl+A 전체)")
        act(emoji_icon("✏️"), "이름변경", self.rename_current,
            "현재 사진 파일명 직접 변경 (F2)")
        tb.addSeparator()
        # 3) 조정 도구 (일회성 + 토글 혼합)
        act(SP.SP_BrowserReload, "자동 재인식", self.redetect, "이 사진에서 사각형 다시 찾기")
        act(SP.SP_FileDialogListView, "화면에 맞춤", self.view.reset_fit, "화면에 맞춤 (F/더블클릭)")
        self.act_loupe = QtGui.QAction(std(SP.SP_FileDialogContentsView), "돋보기: 켬", self)
        self.act_loupe.setCheckable(True)
        self.act_loupe.setChecked(True)
        self.act_loupe.setToolTip("모서리 주변 확대 표시 켜기/끄기 (상태 유지 토글)")
        self.act_loupe.toggled.connect(self._loupe_toggled)
        tb.addAction(self.act_loupe)
        tb.addSeparator()
        # 4) 확인·내보내기 (완료 액션 — 내보내기는 강조색)
        self.act_done = QtGui.QAction(std(SP.SP_DialogApplyButton), "수정완료", self)
        self.act_done.setCheckable(True)
        self.act_done.setToolTip(
            "이 사진을 수정완료로 표시 / 다시 누르면 해제 (Enter = 수정완료+다음)")
        self.act_done.toggled.connect(self._done_toggled)
        tb.addAction(self.act_done)
        a_exp = act(SP.SP_DialogSaveButton, "전체 내보내기", self.export_all,
                    "전체를 _scanned/ 폴더에 선택한 형식으로 저장")
        btn = tb.widgetForAction(a_exp)
        if btn is not None:
            btn.setObjectName("primary")

    def _loupe_toggled(self, on: bool):
        self.act_loupe.setText("돋보기: 켬" if on else "돋보기: 끔")
        self.view.set_loupe_enabled(on)

    def _build_options(self) -> QtWidgets.QWidget:
        """보정(사진별) + 출력 형식(전체). 보정은 선택한 사진들에 적용되고
        미리보기·목록 배지에 즉시 반영된다."""
        box = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)

        # ── 보정: 선택한 사진들에 적용(개별) ──
        grp1 = QtWidgets.QGroupBox("이미지 보정 (선택한 사진)")
        f1 = QtWidgets.QVBoxLayout(grp1)
        self.cb_sharpen = QtWidgets.QCheckBox("샤프닝   (1)")
        self.cb_contrast = QtWidgets.QCheckBox("명암 자동 보정   (2)")
        self.cb_gray = QtWidgets.QCheckBox("그레이스케일·흑백   (3)")
        for cb in (self.cb_sharpen, self.cb_contrast, self.cb_gray):
            cb.toggled.connect(self._opt_toggled)
            f1.addWidget(cb)
        outer.addWidget(grp1)

        # ── 출력 형식: 전체 공통 ──
        grp2 = QtWidgets.QGroupBox("출력 형식 (전체 공통)")
        f2 = QtWidgets.QVBoxLayout(grp2)
        self.rb_jpg = QtWidgets.QRadioButton("JPG")
        self.rb_png = QtWidgets.QRadioButton("PNG")
        self.rb_jpg.setChecked(True)
        fmt_group = QtWidgets.QButtonGroup(self)
        fmt_group.addButton(self.rb_jpg)
        fmt_group.addButton(self.rb_png)
        f2.addWidget(self.rb_jpg)
        f2.addWidget(self.rb_png)
        self.rb_jpg.toggled.connect(self._fmt_changed)
        # JPG 품질 (PNG 선택 시 비활성)
        q = QtWidgets.QHBoxLayout()
        self.lbl_q = QtWidgets.QLabel("JPG 품질: 95")
        self.sl_q = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_q.setRange(50, 100)
        self.sl_q.setValue(95)
        self.sl_q.valueChanged.connect(self._quality_changed)
        q.addWidget(self.lbl_q)
        q.addWidget(self.sl_q, 1)
        f2.addLayout(q)
        outer.addWidget(grp2)
        return box

    def _selected_rows(self) -> list[int]:
        """선택된 행(없으면 현재 행)."""
        rows = sorted({ix.row() for ix in self.listw.selectedIndexes()})
        if not rows and 0 <= self.idx < len(self.pages):
            rows = [self.idx]
        return rows

    def _opt_toggled(self):
        """보정 체크박스 토글 → 선택한 사진들 전체에 기록 + 미리보기/배지 갱신."""
        if self._loading:                 # goto 가 체크박스 동기화할 때의 역방향 발화 차단
            return
        s, c, g = (self.cb_sharpen.isChecked(), self.cb_contrast.isChecked(),
                   self.cb_gray.isChecked())
        rows = self._selected_rows()
        for r in rows:
            pg = self.pages[r]
            pg.sharpen, pg.contrast, pg.gray = s, c, g
            self._refresh_item(r)
        self.update_preview()
        self._update_size_estimate()
        if rows:                             # 키 토글 시 어느 옵션이 켜졌는지 피드백
            on = lambda v: "ON" if v else "off"
            self.status.showMessage(
                f"보정 {len(rows)}장 — 샤프닝 {on(s)} · 명암 {on(c)} · 흑백 {on(g)}")

    def _sync_option_checks(self, page: Page):
        """현재 사진의 보정 상태를 체크박스에 반영(역방향 발화 없이)."""
        for cb, val in ((self.cb_sharpen, page.sharpen),
                        (self.cb_contrast, page.contrast),
                        (self.cb_gray, page.gray)):
            cb.blockSignals(True)
            cb.setChecked(val)
            cb.blockSignals(False)

    def _fmt_changed(self, *_):
        on = self.rb_jpg.isChecked()
        self.lbl_q.setEnabled(on)
        self.sl_q.setEnabled(on)
        self._update_size_estimate()

    def _quality_changed(self, v: int):
        self.lbl_q.setText(f"JPG 품질: {v}")
        self._update_size_estimate()

    def _update_size_estimate(self):
        """현재 장의 보정 결과를 축소본으로 인코드해 전체 용량을 근사 추정."""
        if not hasattr(self, "rb_jpg"):
            return
        if self.view.bgr is None:
            self.rb_jpg.setText("JPG")
            self.rb_png.setText("PNG")
            return
        out = self.view.preview_image()
        s2 = self.view._small_scale ** 2
        if out is None or out.size == 0 or s2 <= 0:
            return
        out = enhance(out, **self._enhance_opts())
        ok_j, jb = cv2.imencode(".jpg", out,
                                [cv2.IMWRITE_JPEG_QUALITY, self.sl_q.value()])
        ok_p, pb = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if ok_j:
            self.rb_jpg.setText(f"JPG    (약 {_fmt_size(len(jb) / s2)})")
        if ok_p:
            self.rb_png.setText(f"PNG    (약 {_fmt_size(len(pb) / s2)})")

    def _enhance_opts(self, page: Page | None = None) -> dict:
        """보정 옵션(사진별). page 미지정 시 현재 사진 기준."""
        if page is None and 0 <= self.idx < len(self.pages):
            page = self.pages[self.idx]
        if page is None:
            return dict(sharpen=False, auto_contrast=False, grayscale=False)
        return dict(sharpen=page.sharpen, auto_contrast=page.contrast,
                    grayscale=page.gray)

    def _build_shortcuts(self):
        sc = lambda key, fn: QtGui.QShortcut(QtGui.QKeySequence(key), self, fn)
        sc(QtCore.Qt.Key.Key_Right, self.next)
        sc(QtCore.Qt.Key.Key_Left, self.prev)
        sc(QtCore.Qt.Key.Key_Return, self.confirm)
        sc(QtCore.Qt.Key.Key_Enter, self.confirm)
        sc(QtCore.Qt.Key.Key_F, self.view.reset_fit)
        sc(QtCore.Qt.Key.Key_F2, self.rename_current)
        sc(QtCore.Qt.Key.Key_Delete, self.remove_selected)
        # 보정 옵션 단독키(선택 사진들에 적용) — 체크박스 위→아래와 동일 순서
        sc(QtCore.Qt.Key.Key_1, self.cb_sharpen.toggle)
        sc(QtCore.Qt.Key.Key_2, self.cb_contrast.toggle)
        sc(QtCore.Qt.Key.Key_3, self.cb_gray.toggle)

    # ── 열기 + 사전 인식 ──
    def open_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "사진 열기", "", "이미지 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if paths:
            self._load_paths(paths)

    def open_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "사진 폴더 선택")
        if d:
            self._load_paths(_folder_images(d))

    # ── 드래그앤드랍 (파일/폴더 혼합 허용) ──
    @staticmethod
    def _paths_from_mime(mime: QtCore.QMimeData) -> list[str]:
        out: list[str] = []
        for url in mime.urls():
            p = url.toLocalFile()
            if not p:
                continue
            if os.path.isdir(p):
                out.extend(_folder_images(p))
            elif p.lower().endswith(IMG_EXTS) and os.path.isfile(p):
                out.append(p)
        return out

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls() and self._paths_from_mime(e.mimeData()):
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        paths = self._paths_from_mime(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self._load_paths(paths, append=True)  # 드래그=기존에 추가(교체 아님)

    def _detect_pages(self, pages: list[Page]):
        """주어진 페이지들에 대해 자동 인식·썸네일·라벨 사전 패스(진행창 표시)."""
        prog = QtWidgets.QProgressDialog("자동 인식 중…", "취소", 0, len(pages), self)
        prog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        for i, page in enumerate(pages):
            prog.setValue(i)
            if prog.wasCanceled():
                break
            bgr = None
            try:
                bgr = load_image(page.path)
                page.quad = detect_quad(bgr)
                page.thumb = make_thumb(bgr)
            except Exception:
                page.quad = None
            page.label = read_label(page.path)   # 디코드 성공 여부와 무관하게 라벨 읽기
            del bgr
        prog.setValue(len(pages))

    def _load_paths(self, files: list[str], append: bool = False):
        if not files:
            self.status.showMessage("이미지가 없습니다.")
            return
        if append and self.pages:                # 기존 목록에 추가(이미 있는 경로는 건너뜀)
            existing = {os.path.normcase(p.path) for p in self.pages}
            new = [f for f in files if os.path.normcase(f) not in existing]
            if not new:
                self.status.showMessage("이미 추가된 사진입니다.")
                return
            self._save_current()
            start = len(self.pages)
            pages = [Page(p) for p in new]
            self._detect_pages(pages)
            self.pages.extend(pages)
            self._rebuild_list()
            self.listw.setCurrentRow(start)      # 새로 추가된 첫 장으로 이동
            self.status.showMessage(f"{len(new)}장 추가 — 총 {len(self.pages)}장")
            return
        self.pages = [Page(p) for p in files]
        self.idx = -1
        self._detect_pages(self.pages)
        self._rebuild_list()
        if self.pages:
            self.listw.setCurrentRow(0)

    def _rebuild_list(self):
        self.listw.blockSignals(True)
        self.listw.clear()
        for page in self.pages:
            item = QtWidgets.QListWidgetItem(self._label(page))
            if page.thumb is not None:
                item.setIcon(page.thumb)
            self.listw.addItem(item)
        self.listw.blockSignals(False)
        self._apply_filter()                 # 새 항목에 현재 필터 반영

    def _apply_filter(self):
        """표식 체크박스(OR)에 따라 항목 숨김/표시. pages↔row 1:1 유지, 숨김만 토글."""
        if not hasattr(self, "cb_f_fail"):
            return
        f_fail, f_det, f_conf = (self.cb_f_fail.isChecked(),
                                 self.cb_f_det.isChecked(),
                                 self.cb_f_conf.isChecked())
        for i, page in enumerate(self.pages):
            it = self.listw.item(i)
            if it is None:
                continue
            if page.confirmed:               # 확인됨
                vis = f_conf
            elif page.quad is not None:      # 인식됨(미확인)
                vis = f_det
            else:                            # 미인식
                vis = f_fail
            it.setHidden(not vis)

    def _make_filter_btn(self, text: str, tip: str) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(text)
        b.setCheckable(True)
        b.setChecked(True)                   # 기본 켬(=전체 표시)
        b.setToolTip(tip)
        b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        b.toggled.connect(self._apply_filter)  # setChecked 뒤 연결(초기 발화 방지)
        return b

    def select_visible(self):
        """현재 필터로 보이는 항목만 전체 선택(전체선택 버튼)."""
        self.listw.clearSelection()
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            if not it.isHidden():
                it.setSelected(True)

    def _toggle_select_all(self, on: bool):
        """전체 선택 ↔ 전체 해제 토글."""
        if on:
            self.select_visible()
            self.btn_all.setText("전체 해제")
        else:
            self.listw.clearSelection()
            self.btn_all.setText("전체 선택")

    def _label(self, page: Page) -> str:
        """목록 항목 HTML(HtmlDelegate 가 렌더). 파일명/태그/배지를 글꼴로 구분."""
        mark = "✓" if page.confirmed else ("•" if page.quad is not None else "?")
        name = html.escape(os.path.basename(page.path))
        parts = [f'<div style="color:#f0f0f0;font-weight:bold;">{mark} {name}</div>']
        if page.label:                       # EXIF 라벨 — 기울임·옅은 파랑·작게로 구분
            parts.append(
                '<div style="color:#6fb7ff;font-style:italic;font-size:11px;">'
                f'🏷 {html.escape(page.label)}</div>')
        badges = ([] + (["✦샤프"] if page.sharpen else [])
                  + (["◐명암"] if page.contrast else [])
                  + (["⬛흑백"] if page.gray else []))
        if badges:                           # 켜진 보정 배지(한눈에 비교용)
            parts.append(
                '<div style="color:#9a9a9a;font-size:11px;">'
                f'{" ".join(badges)}</div>')
        return "".join(parts)

    def _refresh_item(self, i: int):
        self.listw.item(i).setText(self._label(self.pages[i]))

    def _list_menu(self, pos: QtCore.QPoint):
        """썸네일 항목 우클릭 메뉴(상단바와 겸용 동작)."""
        item = self.listw.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():            # 선택 밖 항목 우클릭 → 그 항목을 현재로
            self.listw.setCurrentItem(item)
        std = self.style().standardIcon
        SP = QtWidgets.QStyle.StandardPixmap
        row = self.listw.row(item)
        will_confirm = not self.pages[row].confirmed  # 클릭 항목 기준으로 토글 방향 결정
        menu = QtWidgets.QMenu(self)
        a_done = menu.addAction(
            std(SP.SP_DialogApplyButton),
            "수정완료" if will_confirm else "수정완료 해제")
        menu.addSeparator()
        a_lbl = menu.addAction(emoji_icon("🏷"), "라벨로 이름변경")
        a_ren = menu.addAction(emoji_icon("✏️"), "이름변경 (F2)")
        menu.addSeparator()
        a_det = menu.addAction(std(SP.SP_BrowserReload), "자동 재인식")
        menu.addSeparator()
        a_rm = menu.addAction(std(SP.SP_TrashIcon), "목록에서 제거 (Del)")
        chosen = menu.exec(self.listw.viewport().mapToGlobal(pos))
        if chosen == a_done:
            self._set_confirmed(self._selected_rows() or [row], will_confirm)
        elif chosen == a_lbl:
            self.rename_to_label()
        elif chosen == a_ren:
            self.rename_current()
        elif chosen == a_det:
            self.redetect()
        elif chosen == a_rm:
            self.remove_selected()

    # ── 미리보기 ──
    def update_preview(self):
        out = self.view.preview_image() if self.view.bgr is not None else None
        if out is None or out.size == 0:
            return
        out = enhance(out, **self._enhance_opts())  # 보정 옵션 즉시 반영
        self.preview.set_source(bgr_to_qpixmap(out))  # 크기 추종은 라벨이 알아서

    # ── 네비게이션 ──
    def _save_current(self):
        if 0 <= self.idx < len(self.pages) and self.view.bgr is not None:
            self.pages[self.idx].quad = self.view.quad()

    def goto(self, row: int):
        if self._loading or row < 0 or row >= len(self.pages):
            return
        self._save_current()
        self._loading = True
        try:
            self.idx = row
            page = self.pages[row]
            bgr = load_image(page.path)
            self.view.set_image(bgr)
            h, w = bgr.shape[:2]
            self.view.set_quad(page.quad if page.quad is not None
                               else default_quad(w, h))
            self._sync_option_checks(page)   # 체크박스를 이 사진의 보정 상태로
            self._sync_done_action()         # 수정완료 토글 상태 동기화
            # (setCurrentRow 재호출 금지: ExtendedSelection 에서 다중선택을 지워버림.
            #  goto 는 current 가 이미 바뀐 뒤 호출되는 슬롯이라 불필요.)
            self.status.showMessage(
                f"{row + 1} / {len(self.pages)}  —  {os.path.basename(page.path)}"
                f"  ({w}x{h})" + ("   [수정완료]" if page.confirmed else ""))
            self._update_size_estimate()
        finally:
            self._loading = False

    def next(self):
        r = self.idx + 1
        while r < len(self.pages) and self.listw.item(r) is not None \
                and self.listw.item(r).isHidden():
            r += 1                            # 필터로 숨겨진 항목은 건너뜀
        if r < len(self.pages):
            self.listw.setCurrentRow(r)

    def prev(self):
        r = self.idx - 1
        while r >= 0 and self.listw.item(r) is not None \
                and self.listw.item(r).isHidden():
            r -= 1
        if r >= 0:
            self.listw.setCurrentRow(r)

    def remove_selected(self):
        """선택한(없으면 현재) 사진을 목록에서 제거. 디스크 파일은 건드리지 않음."""
        rows = self._selected_rows()
        if not rows:
            return
        self._save_current()
        for r in reversed(rows):
            del self.pages[r]
        self._rebuild_list()
        if not self.pages:
            self.idx = -1
            self.view.bgr = None
            self.view.scene().clear()
            self.view.handles = []
            self.view._place_nav()
            self.preview._src = None
            self.preview.setText("(사진을 여세요)")
            self.status.showMessage("목록이 비었습니다.")
            return
        self.idx = -1                          # goto 가 새로 로드하도록
        self.listw.setCurrentRow(min(rows[0], len(self.pages) - 1))
        self.status.showMessage(f"{len(rows)}장 제거 — 남은 {len(self.pages)}장")

    def redetect(self):
        if self.view.bgr is None:
            return
        q = detect_quad(self.view.bgr)
        if q is None:
            self.status.showMessage("자동 인식 실패 — 클릭/드래그로 맞추세요.")
        else:
            self.view.set_quad(q)

    # ── 이름 변경 ──
    def _rename_file(self, page: Page, newstem: str) -> bool:
        """page 의 실제 파일을 newstem 으로 rename. 중복명은 ' (2)' 식 회피.
        성공 시 page.path 갱신 후 True, 변화 없음/실패 시 False."""
        stem = safe_stem(newstem)
        d = os.path.dirname(page.path)
        ext = os.path.splitext(page.path)[1]
        target = os.path.join(d, stem + ext)
        if os.path.normcase(target) == os.path.normcase(page.path):
            return False                          # 이미 그 이름 → 변화 없음
        i = 2
        while os.path.exists(target):
            target = os.path.join(d, f"{stem} ({i}){ext}")
            i += 1
        try:
            os.rename(page.path, target)
        except OSError as e:
            print("이름변경 실패:", page.path, e)
            return False
        page.path = target
        return True

    def rename_to_label(self):
        """선택된(없으면 현재) 사진들의 파일명을 각자의 EXIF 라벨로 일괄 변경."""
        rows = sorted({ix.row() for ix in self.listw.selectedIndexes()})
        if not rows and 0 <= self.idx < len(self.pages):
            rows = [self.idx]
        labeled = [r for r in rows if self.pages[r].label]
        if not labeled:
            self.status.showMessage("선택한 사진에 EXIF 라벨이 없습니다.")
            return
        self._save_current()
        n = sum(self._rename_file(self.pages[r], self.pages[r].label) for r in labeled)
        self._rebuild_list()
        if 0 <= self.idx < len(self.pages):
            self.listw.setCurrentRow(self.idx)
        self.status.showMessage(f"라벨로 이름변경: {n} / {len(labeled)}장")

    def rename_current(self):
        """현재 사진 파일명을 직접 입력해 변경(F2)."""
        if not (0 <= self.idx < len(self.pages)):
            return
        page = self.pages[self.idx]
        cur = os.path.splitext(os.path.basename(page.path))[0]
        text, ok = QtWidgets.QInputDialog.getText(
            self, "이름 변경", "새 파일명 (확장자 제외):",
            QtWidgets.QLineEdit.EchoMode.Normal, cur)
        if not ok or not text.strip():
            return
        self._save_current()
        if self._rename_file(page, text):
            self._rebuild_list()
            self.listw.setCurrentRow(self.idx)
            self.status.showMessage(f"이름변경 → {os.path.basename(page.path)}")

    def confirm(self):
        """Enter: 현재 사진 수정완료 표시 + 다음 장(빠른 처리 경로)."""
        if self.idx < 0 or self.view.bgr is None:
            return
        self._save_current()
        self.pages[self.idx].confirmed = True
        self._refresh_item(self.idx)
        self._apply_filter()                 # 수정완료 표식 바뀜 → 필터 재반영
        self._sync_done_action()
        self.next()

    def _set_confirmed(self, rows: list[int], value: bool):
        """선택 사진들의 수정완료 상태를 일괄 설정(우클릭 메뉴/토글 공용)."""
        if not rows:
            return
        self._save_current()
        for r in rows:
            self.pages[r].confirmed = value
            self._refresh_item(r)
        self._apply_filter()
        self._sync_done_action()

    def _done_toggled(self, on: bool):
        """툴바 '수정완료' 토글 → 현재 사진 상태 설정(해제 가능). goto 동기화는 무시."""
        if self._loading or not (0 <= self.idx < len(self.pages)):
            return
        self._set_confirmed([self.idx], on)

    def _sync_done_action(self):
        """현재 사진의 수정완료 여부를 툴바 토글에 반영(역방향 발화 없이)."""
        if not hasattr(self, "act_done"):
            return
        on = 0 <= self.idx < len(self.pages) and self.pages[self.idx].confirmed
        self.act_done.blockSignals(True)
        self.act_done.setChecked(bool(on))
        self.act_done.blockSignals(False)

    # ── 내보내기 ──
    def export_all(self):
        if not self.pages:
            return
        self._save_current()
        prog = QtWidgets.QProgressDialog("내보내는 중…", "취소", 0, len(self.pages), self)
        prog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        done = 0
        for i, page in enumerate(self.pages):
            prog.setValue(i)
            if prog.wasCanceled():
                break
            bgr = None
            try:
                bgr = load_image(page.path)
                h, w = bgr.shape[:2]
                quad = page.quad if page.quad is not None else default_quad(w, h)
                out = enhance(warp(bgr, quad), **self._enhance_opts(page))
                out_dir = os.path.join(os.path.dirname(page.path), "_scanned")
                os.makedirs(out_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(page.path))[0]
                if self.rb_jpg.isChecked():
                    save_image(os.path.join(out_dir, stem + ".jpg"), out,
                               quality=self.sl_q.value())
                else:
                    save_image(os.path.join(out_dir, stem + ".png"), out)
                done += 1
            except Exception as e:
                print("내보내기 실패:", page.path, e)
            del bgr
        prog.setValue(len(self.pages))
        self.status.showMessage(f"내보내기 완료: {done} / {len(self.pages)}")
        if QtGui.QGuiApplication.platformName() != "offscreen":  # selftest 에선 모달 금지
            QtWidgets.QMessageBox.information(
                self, "완료",
                f"내보내기 완료: {done} / {len(self.pages)}\n→ 각 사진 옆 _scanned/ 폴더")


def apply_dark(app: QtWidgets.QApplication):
    """앱 전역 다크 테마: Fusion 스타일 + 다크 팔레트(대화상자·목록·진행창 포함)."""
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    C = QtGui.QColor
    R = QtGui.QPalette.ColorRole
    p.setColor(R.Window, C(43, 43, 43))
    p.setColor(R.WindowText, C(224, 224, 224))
    p.setColor(R.Base, C(30, 30, 30))
    p.setColor(R.AlternateBase, C(40, 40, 40))
    p.setColor(R.Text, C(224, 224, 224))
    p.setColor(R.Button, C(58, 58, 58))
    p.setColor(R.ButtonText, C(224, 224, 224))
    p.setColor(R.Highlight, C(45, 127, 249))
    p.setColor(R.HighlightedText, C(255, 255, 255))
    p.setColor(R.ToolTipBase, C(50, 50, 50))
    p.setColor(R.ToolTipText, C(224, 224, 224))
    p.setColor(R.PlaceholderText, C(140, 140, 140))
    dis = QtGui.QPalette.ColorGroup.Disabled
    p.setColor(dis, R.Text, C(120, 120, 120))
    p.setColor(dis, R.ButtonText, C(120, 120, 120))
    app.setPalette(p)


def _selftest(folder: str) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    apply_dark(app)
    w = MainWindow()
    w.resize(1400, 880)
    w.show()
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(IMG_EXTS))
    w._load_paths(files)
    assert w.pages and w.idx == 0, "로드 실패"
    n = len(w.pages)
    assert all(p.thumb is not None for p in w.pages), "썸네일 생성 실패"
    # 미리보기 산출
    assert w.view.preview_image() is not None, "미리보기 실패"
    # 클릭=가장 가까운 꼭짓점 이동
    from PySide6.QtCore import QPointF
    w.view.place_nearest(QPointF(10, 10))
    assert tuple(round(v) for v in w.view.quad()[0]) == (10, 10), "클릭-코너 실패"
    # 확인 → 다음
    w.confirm()
    assert w.pages[0].confirmed, "확인 실패"
    # 용량 추정 표시 확인
    assert "약" in w.rb_jpg.text(), "용량 추정 표시 실패"
    # 내보내기(JPG 기본 → 장당 1파일)
    w.export_all()
    out_dir = os.path.join(folder, "_scanned")
    assert os.path.isdir(out_dir) and len(os.listdir(out_dir)) == n, "내보내기 부족"
    assert all(f.endswith(".jpg") for f in os.listdir(out_dir)), "JPG 형식 아님"
    print(f"selftest 통과: 페이지 {n}, 검출 {sum(p.quad is not None for p in w.pages)}")
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(_selftest(sys.argv[2]))
    app = QtWidgets.QApplication(sys.argv)
    apply_dark(app)
    w = MainWindow()
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            w._load_paths(_folder_images(arg))
        elif os.path.isfile(arg):
            w._load_paths([arg])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
