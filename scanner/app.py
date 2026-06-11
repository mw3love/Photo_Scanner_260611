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

import os
import sys

import numpy as np
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from scanner.engine import load_image, detect_quad, warp, save_image, enhance

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

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
        """잡기(드래그) 영역을 두 팔에만 한정. 꼭짓점 중심은 비워, 그 지점을
        클릭하면 '가장 가까운 꼭짓점 이동'이 동작하게 한다(정밀 클릭 복원)."""
        path = QtGui.QPainterPath()
        dx, dy = self.DIRS[self.idx]
        pad = 8.0
        x0, x1 = sorted((self.GAP * dx, self.ARM * dx))
        path.addRect(QtCore.QRectF(x0, -pad, x1 - x0, 2 * pad))      # 가로 팔
        y0, y1 = sorted((self.GAP * dy, self.ARM * dy))
        path.addRect(QtCore.QRectF(-pad, y0, 2 * pad, y1 - y0))      # 세로 팔
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

        self.on_quad_changed = lambda: None        # 윈도우가 주입(미리보기 갱신)
        self.loupe_enabled = True
        self._loupe = QtWidgets.QLabel(self.viewport())
        self._loupe.setFixedSize(180, 180)
        self._loupe.setStyleSheet(
            "border:2px solid #222; background:#282828;")
        self._loupe.hide()

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
            self._active = self._nearest(self.mapToScene(e.position().toPoint()))
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
        c = self.handles[self._active].pos()
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


class Page:
    __slots__ = ("path", "quad", "confirmed", "thumb")

    def __init__(self, path: str):
        self.path = path
        self.quad: np.ndarray | None = None
        self.confirmed = False
        self.thumb: QtGui.QIcon | None = None


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
        self.listw.currentRowChanged.connect(self.goto)
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

        split = QtWidgets.QSplitter()
        split.addWidget(self.listw)
        split.addWidget(self.view)
        split.addWidget(prev_box)
        split.setStretchFactor(1, 3)  # 창 확대 시 캔버스:미리보기 = 3:1 로 분배
        split.setStretchFactor(2, 1)
        split.setSizes([200, 880, 320])

        # 하단 조작 안내바
        self.helpbar = QtWidgets.QLabel(
            "  F: 맞춤   ·   Enter: 확인+다음   ·   ← →: 이전/다음")
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

    def _build_toolbar(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.setStyleSheet(self.TOOLBAR_QSS)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        std = self.style().standardIcon
        SP = QtWidgets.QStyle.StandardPixmap

        def act(icon, text, slot, tip=""):
            a = QtGui.QAction(std(icon), text, self)
            a.triggered.connect(slot)
            if tip:
                a.setToolTip(tip)
            tb.addAction(a)
            return a

        # 1) 열기 (일회성: 대화상자 팝업)
        act(SP.SP_FileDialogStart, "사진 열기", self.open_files, "사진 파일 선택(여러 장 가능)")
        act(SP.SP_DirOpenIcon, "폴더 열기", self.open_folder, "폴더 안 이미지 전부 열기")
        tb.addSeparator()
        # 2) 탐색 (일회성: 즉시 동작)
        act(SP.SP_ArrowLeft, "이전", self.prev, "이전 사진 (←)")
        act(SP.SP_ArrowRight, "다음", self.next, "다음 사진 (→)")
        tb.addSeparator()
        # 3) 조정 도구 (일회성 + 토글 혼합)
        act(SP.SP_BrowserReload, "자동 재인식", self.redetect, "이 사진에서 사각형 다시 찾기")
        act(SP.SP_FileDialogListView, "맞춤", self.view.reset_fit, "화면에 맞춤 (F/더블클릭)")
        self.act_loupe = QtGui.QAction(std(SP.SP_FileDialogContentsView), "돋보기: 켬", self)
        self.act_loupe.setCheckable(True)
        self.act_loupe.setChecked(True)
        self.act_loupe.setToolTip("모서리 주변 확대 표시 켜기/끄기 (상태 유지 토글)")
        self.act_loupe.toggled.connect(self._loupe_toggled)
        tb.addAction(self.act_loupe)
        tb.addSeparator()
        # 4) 확인·내보내기 (완료 액션 — 내보내기는 강조색)
        act(SP.SP_DialogApplyButton, "이 장 확인", self.confirm, "현재 모서리 확정 후 다음 장 (Enter)")
        a_exp = act(SP.SP_DialogSaveButton, "전체 내보내기", self.export_all,
                    "전체를 _scanned/ 폴더에 선택한 형식으로 저장")
        btn = tb.widgetForAction(a_exp)
        if btn is not None:
            btn.setObjectName("primary")

    def _loupe_toggled(self, on: bool):
        self.act_loupe.setText("돋보기: 켬" if on else "돋보기: 끔")
        self.view.set_loupe_enabled(on)

    def _build_options(self) -> QtWidgets.QGroupBox:
        """보정 품질 + 출력 형식 옵션. 보정 항목은 미리보기에 즉시 반영."""
        grp = QtWidgets.QGroupBox("내보내기 옵션")
        form = QtWidgets.QVBoxLayout(grp)
        self.cb_sharpen = QtWidgets.QCheckBox("샤프닝")
        self.cb_contrast = QtWidgets.QCheckBox("명암 자동 보정")
        self.cb_gray = QtWidgets.QCheckBox("그레이스케일(흑백)")
        for cb in (self.cb_sharpen, self.cb_contrast, self.cb_gray):
            cb.toggled.connect(self.update_preview)
            cb.toggled.connect(self._update_size_estimate)
            form.addWidget(cb)
        # 출력 형식 (둘 중 하나, 용량은 현재 장 기준 근사치)
        form.addWidget(QtWidgets.QLabel("출력 형식 (하나 선택):"))
        self.rb_jpg = QtWidgets.QRadioButton("JPG")
        self.rb_png = QtWidgets.QRadioButton("PNG")
        self.rb_jpg.setChecked(True)
        fmt_group = QtWidgets.QButtonGroup(self)
        fmt_group.addButton(self.rb_jpg)
        fmt_group.addButton(self.rb_png)
        form.addWidget(self.rb_jpg)
        form.addWidget(self.rb_png)
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
        form.addLayout(q)
        return grp

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

    def _enhance_opts(self) -> dict:
        return dict(sharpen=self.cb_sharpen.isChecked(),
                    auto_contrast=self.cb_contrast.isChecked(),
                    grayscale=self.cb_gray.isChecked())

    def _build_shortcuts(self):
        sc = lambda key, fn: QtGui.QShortcut(QtGui.QKeySequence(key), self, fn)
        sc(QtCore.Qt.Key.Key_Right, self.next)
        sc(QtCore.Qt.Key.Key_Left, self.prev)
        sc(QtCore.Qt.Key.Key_Return, self.confirm)
        sc(QtCore.Qt.Key.Key_Enter, self.confirm)
        sc(QtCore.Qt.Key.Key_F, self.view.reset_fit)

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
            self._load_paths(paths)

    def _load_paths(self, files: list[str]):
        if not files:
            self.status.showMessage("이미지가 없습니다.")
            return
        self.pages = [Page(p) for p in files]
        self.idx = -1
        prog = QtWidgets.QProgressDialog("자동 인식 중…", "취소", 0, len(self.pages), self)
        prog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        for i, page in enumerate(self.pages):
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
            del bgr
        prog.setValue(len(self.pages))
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

    def _label(self, page: Page) -> str:
        mark = "✓ " if page.confirmed else ("• " if page.quad is not None else "? ")
        return mark + os.path.basename(page.path)

    def _refresh_item(self, i: int):
        self.listw.item(i).setText(self._label(self.pages[i]))

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
            self.listw.setCurrentRow(row)
            self.status.showMessage(
                f"{row + 1} / {len(self.pages)}  —  {os.path.basename(page.path)}"
                f"  ({w}x{h})" + ("   [확인됨]" if page.confirmed else ""))
            self._update_size_estimate()
        finally:
            self._loading = False

    def next(self):
        if self.idx + 1 < len(self.pages):
            self.listw.setCurrentRow(self.idx + 1)

    def prev(self):
        if self.idx > 0:
            self.listw.setCurrentRow(self.idx - 1)

    def redetect(self):
        if self.view.bgr is None:
            return
        q = detect_quad(self.view.bgr)
        if q is None:
            self.status.showMessage("자동 인식 실패 — 클릭/드래그로 맞추세요.")
        else:
            self.view.set_quad(q)

    def confirm(self):
        if self.idx < 0 or self.view.bgr is None:
            return
        self.pages[self.idx].quad = self.view.quad()
        self.pages[self.idx].confirmed = True
        self._refresh_item(self.idx)
        self.next()

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
                out = enhance(warp(bgr, quad), **self._enhance_opts())
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
