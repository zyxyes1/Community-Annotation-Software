import os
import sys
import numpy as np
import io
import tempfile
import hashlib
import json
import math
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QFileDialog, QListWidget,
                            QListWidgetItem, QLabel, QScrollArea, QMessageBox,
                            QProgressDialog, QAction, QMenuBar, QToolBar, QStatusBar,
                            QCheckBox, QGroupBox, QLineEdit, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QWheelEvent, QMouseEvent, QPen, QColor, QFont, QPolygonF
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QSize, QRect, QTimer, QLineF, QDateTime, QPointF

try:
    from osgeo import gdal, osr
    gdal_available = True
except ImportError:
    gdal_available = False
    from PIL import Image
    try:
        import tifffile
        tifffile_available = True
    except ImportError:
        tifffile_available = False

if not gdal_available:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

# ==================== Convex Hull ====================
def convex_hull(points):
    if len(points) <= 1:
        return points
    pts = []
    for p in points:
        if isinstance(p, QPointF):
            pts.append((p.x(), p.y()))
        elif isinstance(p, QPoint):
            pts.append((p.x(), p.y()))
        else:
            pts.append((p[0], p[1]))
    pts = sorted(pts)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) == 2 and hull[0] == hull[1]:
        hull = hull[:1]
    return [QPointF(x, y) for x, y in hull]

# ==================== GeoCoordinateConverter ====================
class GeoCoordinateConverter:
    def __init__(self, geo_transform=None, projection=None):
        self.geo_transform = geo_transform
        self.projection = projection
        self.has_geo_info = geo_transform is not None and projection is not None
        if self.has_geo_info:
            try:
                self.source_srs = osr.SpatialReference()
                self.source_srs.ImportFromWkt(projection)
                self.target_srs = osr.SpatialReference()
                self.target_srs.ImportFromEPSG(4326)
                self.transform_to_latlon = osr.CoordinateTransformation(self.source_srs, self.target_srs)
                self.transform_from_latlon = osr.CoordinateTransformation(self.target_srs, self.source_srs)
            except Exception as e:
                print(f"Failed to initialize coordinate converter: {str(e)}")
                self.has_geo_info = False
    
    def pixel_to_latlon(self, x, y):
        if not self.has_geo_info or not self.geo_transform:
            return None
        try:
            geo_x = self.geo_transform[0] + x * self.geo_transform[1] + y * self.geo_transform[2]
            geo_y = self.geo_transform[3] + x * self.geo_transform[4] + y * self.geo_transform[5]
            latlon = self.transform_to_latlon.TransformPoint(geo_x, geo_y)
            return (latlon[1], latlon[0])
        except Exception as e:
            print(f"Pixel to lat/lon failed: {str(e)}")
            return None
    
    def latlon_to_pixel(self, lon, lat):
        if not self.has_geo_info or not self.geo_transform:
            return None
        try:
            point = self.transform_from_latlon.TransformPoint(lon, lat)
            geo_x, geo_y = point[0], point[1]
            det = self.geo_transform[1] * self.geo_transform[5] - self.geo_transform[2] * self.geo_transform[4]
            if det == 0:
                return None
            x = (self.geo_transform[5] * (geo_x - self.geo_transform[0]) -
                 self.geo_transform[2] * (geo_y - self.geo_transform[3])) / det
            y = (-self.geo_transform[4] * (geo_x - self.geo_transform[0]) +
                 self.geo_transform[1] * (geo_y - self.geo_transform[3])) / det
            return (x, y)
        except Exception as e:
            print(f"Lat/lon to pixel failed: {str(e)}")
            return None

# ==================== MultiLevelCacheManager ====================
class MultiLevelCacheManager:
    def __init__(self):
        self.cache_dir = tempfile.mkdtemp(prefix="tiff_cache_")
        self.cache_levels = {20.0: "level_20x", 10.0: "level_10x", 5.0: "level_5x",
                             2.5: "level_2.5x", 1.0: "level_1x"}
        self.current_levels = {}
        self.file_hash = ""
        self.file_cache_dir = None
    
    def set_file(self, file_path):
        with open(file_path, 'rb') as f:
            self.file_hash = hashlib.md5(f.read()).hexdigest()
        self.file_cache_dir = os.path.join(self.cache_dir, self.file_hash)
        if not os.path.exists(self.file_cache_dir):
            os.makedirs(self.file_cache_dir)
        self.current_levels = {}
    
    def get_cache_path(self, level):
        if level not in self.cache_levels or self.file_cache_dir is None:
            return None
        return os.path.join(self.file_cache_dir, f"{self.cache_levels[level]}.jpg")
    
    def is_level_cached(self, level):
        cache_path = self.get_cache_path(level)
        return cache_path and os.path.exists(cache_path)
    
    def save_level(self, level, qimage):
        cache_path = self.get_cache_path(level)
        if cache_path and not qimage.isNull():
            qimage.save(cache_path, "JPEG", quality=85)
            self.current_levels[level] = cache_path
            return True
        return False
    
    def load_level(self, level):
        cache_path = self.get_cache_path(level)
        if cache_path and os.path.exists(cache_path):
            return QPixmap(cache_path)
        return None
    
    def cleanup(self):
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
        except:
            pass

# ==================== TIFFAnalyzer ====================
class TIFFAnalyzer(QThread):
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            info = {}
            if gdal_available:
                dataset = gdal.Open(self.file_path)
                if not dataset:
                    self.error_occurred.emit(f"Unable to open file: {self.file_path}")
                    return
                info['width'] = dataset.RasterXSize
                info['height'] = dataset.RasterYSize
                info['bands'] = dataset.RasterCount
                info['driver'] = dataset.GetDriver().ShortName
                band = dataset.GetRasterBand(1)
                info['data_type'] = gdal.GetDataTypeName(band.DataType)
                info['has_pyramids'] = len(dataset.GetSubDatasets()) > 0
                metadata = dataset.GetMetadata()
                info['metadata'] = {k: v for k, v in metadata.items() if not k.startswith('_')}
                geo_transform = dataset.GetGeoTransform()
                projection = dataset.GetProjection()
                if geo_transform and geo_transform != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                    info['geo_transform'] = list(geo_transform)
                    info['projection'] = projection
                else:
                    info['geo_transform'] = None
                    info['projection'] = None
                dataset = None
            else:
                if not tifffile_available:
                    self.error_occurred.emit("GDAL or tifffile library not found, cannot analyze TIFF")
                    return
                with tifffile.TiffFile(self.file_path) as tif:
                    page = tif.pages[0]
                    info['width'] = page.shape[1] if len(page.shape) > 1 else page.shape[0]
                    info['height'] = page.shape[0] if len(page.shape) > 1 else 1
                    info['bands'] = page.shape[2] if len(page.shape) > 2 else 1
                    info['data_type'] = str(page.dtype)
                    info['driver'] = 'TIFF'
                    info['has_pyramids'] = len(tif.pages) > 1
                    info['geo_transform'] = None
                    info['projection'] = None
                    info['metadata'] = {}
                    if hasattr(page, 'tags'):
                        for tag in page.tags.values():
                            if hasattr(tag, 'value') and not tag.name.startswith('_'):
                                info['metadata'][tag.name] = str(tag.value)
            self.analysis_complete.emit(info)
        except Exception as e:
            self.error_occurred.emit(f"Error analyzing TIFF: {str(e)}")

# ==================== LevelCacheGenerator ====================
class LevelCacheGenerator(QThread):
    progress_updated = pyqtSignal(int, float)
    level_cached = pyqtSignal(float, QPixmap)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, cache_manager, target_level):
        super().__init__()
        self.file_path = file_path
        self.cache_manager = cache_manager
        self.target_level = target_level
        self.running = True

    @staticmethod
    def replace_black_with_white(rgb_array):
        mask = (rgb_array[:, :, 0] == 0) & (rgb_array[:, :, 1] == 0) & (rgb_array[:, :, 2] == 0)
        rgb_array[mask] = [255, 255, 255]
        return rgb_array

    def run(self):
        try:
            if not self.running:
                return
            self.progress_updated.emit(10, self.target_level)
            if gdal_available:
                dataset = gdal.Open(self.file_path)
                if not dataset:
                    self.error_occurred.emit(f"Unable to open file: {self.file_path}")
                    return
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                bands = dataset.RasterCount
                self.progress_updated.emit(20, self.target_level)
                new_width = int(width / self.target_level)
                new_height = int(height / self.target_level)
                if new_width < 1: new_width = 1
                if new_height < 1: new_height = 1

                data = []
                for i in range(min(bands, 3)):
                    if not self.running:
                        return
                    band = dataset.GetRasterBand(i + 1)
                    band_data = band.ReadAsArray(0, 0, width, height, new_width, new_height)
                    data.append(band_data)
                    self.progress_updated.emit(20 + (i + 1) * 60 // min(bands, 3), self.target_level)

                if len(data) == 1:
                    gray = data[0]
                    if gray.dtype == np.float32 or gray.dtype == np.float64:
                        min_val = gray.min()
                        max_val = gray.max()
                        if max_val > min_val:
                            gray = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            gray = np.zeros_like(gray, dtype=np.uint8)
                    else:
                        gray = gray.astype(np.uint8)
                    rgb_array = np.stack((gray, gray, gray), axis=-1)
                elif len(data) >= 3:
                    r, g, b = data[0], data[1], data[2]
                    for i in range(3):
                        min_val = data[i].min()
                        max_val = data[i].max()
                        if max_val > min_val:
                            data[i] = ((data[i] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            data[i] = np.zeros_like(data[i], dtype=np.uint8)
                    rgb_array = np.stack((data[0], data[1], data[2]), axis=-1)
                else:
                    self.error_occurred.emit("Image has fewer than 1 band, cannot process")
                    return

                rgb_array = self.replace_black_with_white(rgb_array)
                q_image = QImage(rgb_array.data, new_width, new_height, new_width * 3, QImage.Format_RGB888)

            else:
                if not tifffile_available:
                    self.error_occurred.emit("GDAL or tifffile not found, cannot load TIFF")
                    return
                try:
                    with tifffile.TiffFile(self.file_path) as tif:
                        img_array = tif.pages[0].asarray()
                        self.progress_updated.emit(30, self.target_level)
                        if len(img_array.shape) == 2:
                            height, width = img_array.shape
                            bands = 1
                        elif len(img_array.shape) == 3:
                            height, width, bands = img_array.shape
                        else:
                            self.error_occurred.emit(f"Unsupported image dimensions: {len(img_array.shape)}")
                            return
                        self.progress_updated.emit(50, self.target_level)
                        new_width = int(width / self.target_level)
                        new_height = int(height / self.target_level)
                        if new_width < 1: new_width = 1
                        if new_height < 1: new_height = 1

                        if bands == 1:
                            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                                min_val = img_array.min()
                                max_val = img_array.max()
                                if max_val > min_val:
                                    img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                                else:
                                    img_array = np.zeros_like(img_array, dtype=np.uint8)
                                pil_img = Image.fromarray(img_array, mode='L')
                            else:
                                pil_img = Image.fromarray(img_array).convert('L')
                            pil_img = pil_img.convert('RGB')
                        else:
                            if bands >= 3:
                                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                                    rgb_array = img_array[:,:,:3].copy()
                                    for i in range(3):
                                        min_val = rgb_array[:,:,i].min()
                                        max_val = rgb_array[:,:,i].max()
                                        if max_val > min_val:
                                            rgb_array[:,:,i] = ((rgb_array[:,:,i] - min_val) / (max_val - min_val) * 255)
                                        else:
                                            rgb_array[:,:,i] = 0
                                    rgb_array = rgb_array.astype(np.uint8)
                                    pil_img = Image.fromarray(rgb_array, mode='RGB')
                                else:
                                    pil_img = Image.fromarray(img_array[:,:,:3]).convert('RGB')
                            else:
                                pil_img = Image.fromarray(img_array).convert('RGB')

                        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                        self.progress_updated.emit(80, self.target_level)

                        rgb_array = np.array(pil_img)
                        rgb_array = self.replace_black_with_white(rgb_array)
                        pil_img = Image.fromarray(rgb_array)

                        img_byte_arr = io.BytesIO()
                        pil_img.save(img_byte_arr, format='JPEG', quality=85)
                        img_byte_arr = img_byte_arr.getvalue()
                        q_image = QImage.fromData(img_byte_arr)
                except Exception as e:
                    self.error_occurred.emit(f"Fallback loading failed: {str(e)}")
                    return

            self.progress_updated.emit(95, self.target_level)
            if self.running and not q_image.isNull():
                self.cache_manager.save_level(self.target_level, q_image)
                pixmap = QPixmap.fromImage(q_image)
                self.level_cached.emit(self.target_level, pixmap)
            self.progress_updated.emit(100, self.target_level)
        except Exception as e:
            if self.running:
                self.error_occurred.emit(f"Error generating cache level {self.target_level}: {str(e)}")
    
    def stop(self):
        self.running = False
        self.wait()

# ==================== ImageDisplayLabel ====================
class ImageDisplayLabel(QLabel):
    cache_requested = pyqtSignal(float)
    point_added = pyqtSignal(QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: white;")
        self.current_pixmap = None
        self.cache_manager = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.dragging = False
        self.last_pos = QPoint(0, 0)
        self.original_size = (0, 0)
        self.current_file = ""
        self.available_levels = [20.0, 10.0, 5.0, 2.5, 1.0]
        self.current_level_index = 0
        self.current_display_level = 1.0
        self.setMinimumSize(1, 1)
        self.polygons = []
        self.polygon_color = QColor(255, 0, 0)
        self.polygon_line_width = 3
        self.draw_corner_dots = True
        
        # Point-adding mode
        self.add_point_mode = False
        self.fixed_points = []
        self.temp_line = None
    
    def setCacheManager(self, cache_manager):
        self.cache_manager = cache_manager
    
    def setCurrentPixmap(self, pixmap, level=1.0):
        self.current_pixmap = pixmap
        self.current_display_level = level
        if pixmap and not pixmap.isNull():
            self.setMinimumSize(pixmap.size())
        self.update_display()
    
    def setOriginalSize(self, size):
        self.original_size = size
    
    def setCurrentFile(self, file_path):
        self.current_file = file_path
    
    def set_polygons(self, polygon_list):
        self.polygons = polygon_list
        self.update_display()
    
    def clear_polygons(self):
        self.polygons = []
        self.update_display()
    
    def set_add_point_mode(self, enabled):
        self.add_point_mode = enabled
        if not enabled:
            self.fixed_points = []
            self.temp_line = None
        else:
            self.fixed_points = []
            self.temp_line = None
        self.update_display()
    
    def set_fixed_points(self, points):
        self.fixed_points = points
        self.update_display()
    
    def set_temp_line(self, line):
        self.temp_line = line
        self.update_display()
    
    def wheelEvent(self, event):
        pass
    
    def mousePressEvent(self, event):
        if self.add_point_mode and event.button() == Qt.LeftButton:
            orig_pt = self.get_original_pixel_pos(event.pos())
            if orig_pt is not None:
                self.point_added.emit(orig_pt)
            return
        if event.button() == Qt.LeftButton and self.current_pixmap:
            self.dragging = True
            self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        if self.add_point_mode and self.fixed_points:
            orig_pt = self.get_original_pixel_pos(event.pos())
            if orig_pt is not None:
                last_pt = self.fixed_points[-1]
                self.temp_line = QLineF(last_pt, orig_pt)
                self.update_display()
            return
        if self.dragging and self.current_pixmap:
            delta = event.pos() - self.last_pos
            self.offset += delta
            self.last_pos = event.pos()
            self.update_display()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
    
    def get_original_pixel_pos(self, point):
        if not self.current_pixmap:
            return None
        scaled_width = self.current_pixmap.width() * self.scale_factor
        scaled_height = self.current_pixmap.height() * self.scale_factor
        display_x = (self.width() - scaled_width) / 2 + self.offset.x()
        display_y = (self.height() - scaled_height) / 2 + self.offset.y()
        img_x = (point.x() - display_x) / self.scale_factor
        img_y = (point.y() - display_y) / self.scale_factor
        level = self.current_display_level
        if level <= 0:
            level = 1.0
        orig_x = img_x * level
        orig_y = img_y * level
        w, h = self.original_size
        if w > 0 and h > 0:
            orig_x = max(0, min(w-1, orig_x))
            orig_y = max(0, min(h-1, orig_y))
        return QPoint(int(round(orig_x)), int(round(orig_y)))
    
    def update_display(self):
        if not self.current_pixmap:
            return
        canvas = QPixmap(self.size())
        canvas.fill(Qt.white)
        painter = QPainter(canvas)
        scaled_width = self.current_pixmap.width() * self.scale_factor
        scaled_height = self.current_pixmap.height() * self.scale_factor
        display_x = (self.width() - scaled_width) / 2 + self.offset.x()
        display_y = (self.height() - scaled_height) / 2 + self.offset.y()
        scaled_pixmap = self.current_pixmap.scaled(int(scaled_width), int(scaled_height),
                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(int(display_x), int(display_y), scaled_pixmap)
        
        if self.polygons:
            pen = QPen(self.polygon_color)
            pen.setWidth(self.polygon_line_width)
            painter.setPen(pen)
            level = self.current_display_level
            if level <= 0:
                level = 1.0
            for poly in self.polygons:
                if len(poly) < 2:
                    continue
                screen_points = []
                for pt in poly:
                    img_x = pt.x() / level
                    img_y = pt.y() / level
                    img_pt = QPoint(int(round(img_x)), int(round(img_y)))
                    screen_pt = self.transform_screen_point(img_pt)
                    screen_points.append(screen_pt)
                for i in range(len(screen_points)):
                    start = screen_points[i]
                    end = screen_points[(i + 1) % len(screen_points)]
                    painter.drawLine(start, end)
                if self.draw_corner_dots:
                    for pt in screen_points:
                        painter.drawEllipse(pt, 1, 1)
        
        if self.add_point_mode:
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            for pt in self.fixed_points:
                screen_pt = self.transform_screen_point_from_original(pt)
                painter.drawEllipse(screen_pt, 4, 4)
            if self.temp_line and self.fixed_points:
                p1 = self.transform_screen_point_from_original(self.temp_line.p1())
                p2 = self.transform_screen_point_from_original(self.temp_line.p2())
                painter.drawLine(p1, p2)
        
        painter.end()
        super().setPixmap(canvas)
    
    def transform_screen_point(self, point):
        if not self.current_pixmap:
            return point
        scaled_width = self.current_pixmap.width() * self.scale_factor
        scaled_height = self.current_pixmap.height() * self.scale_factor
        display_x = (self.width() - scaled_width) / 2 + self.offset.x()
        display_y = (self.height() - scaled_height) / 2 + self.offset.y()
        screen_x = point.x() * self.scale_factor + display_x
        screen_y = point.y() * self.scale_factor + display_y
        return QPoint(int(screen_x), int(screen_y))
    
    def transform_screen_point_from_original(self, point):
        level = self.current_display_level
        if level <= 0:
            level = 1.0
        cache_pt = QPoint(int(round(point.x() / level)), int(round(point.y() / level)))
        return self.transform_screen_point(cache_pt)
    
    def resizeEvent(self, event):
        if self.current_pixmap:
            self.update_display()
        super().resizeEvent(event)
    
    def reset_view(self):
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        if self.current_pixmap:
            self.update_display()
    
    def zoom_in(self):
        if self.current_pixmap:
            self.scale_factor *= 1.2
            self.update_display()
    
    def zoom_out(self):
        if self.current_pixmap:
            self.scale_factor /= 1.2
            self.update_display()
    
    # Dummy method to satisfy any leftover signal connections
    def request_next_cache(self):
        pass

# ==================== Main Window ====================
class TIFFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.current_file = ""
        self.tiff_analyzer = None
        self.cache_generators = []
        self.cache_manager = MultiLevelCacheManager()
        self.selected_levels = []
        self.cached_levels = []
        self.output_dir = ""
        self.geo_converter = None
        
        self.communities = []
        self.editing_community_index = -1
        self.editing_name = ""
        self.state = "idle"
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('Community Annotation Software')
        self.resize(1400, 900)
        
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        open_folder_action = QAction('Open Folder', self)
        open_folder_action.triggered.connect(self.select_folder)
        file_menu.addAction(open_folder_action)
        open_file_action = QAction('Open File', self)
        open_file_action.triggered.connect(self.select_file)
        file_menu.addAction(open_file_action)
        save_mask_action = QAction('Save Community Mask as TXT', self)
        save_mask_action.triggered.connect(self.save_communities_to_txt)
        file_menu.addAction(save_mask_action)
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu('View')
        reset_view_action = QAction('Reset View', self)
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        
        # ========== Add Help menu with User Manual ==========
        help_menu = menubar.addMenu('Help')
        user_manual_action = QAction('User Manual', self)
        user_manual_action.triggered.connect(self.show_user_manual)
        help_menu.addAction(user_manual_action)
        # ====================================================
        
        toolbar = QToolBar('Toolbar', self)
        self.addToolBar(toolbar)
        toolbar.addAction(open_folder_action)
        toolbar.addAction(open_file_action)
        toolbar.addSeparator()
        toolbar.addAction(reset_view_action)
        toolbar.addSeparator()
        toolbar.addAction(save_mask_action)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Ready')
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        self.folder_path_label = QLabel('No folder selected')
        self.folder_path_label.setWordWrap(True)
        left_layout.addWidget(self.folder_path_label)
        left_layout.addWidget(QLabel('TIFF files:'))
        self.file_list = QListWidget()
        self.file_list.setViewMode(QListWidget.ListMode)
        self.file_list.itemClicked.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)
        self.file_info_label = QLabel('File info will appear here')
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet("background-color: #f8f8f8; padding: 5px; border: 1px solid #ddd;")
        left_layout.addWidget(self.file_info_label)
        
        compression_group = QGroupBox("Select cache levels")
        compression_layout = QVBoxLayout(compression_group)
        self.level_checkboxes = {}
        levels = [(20.0, "20x (downsampled 20x)"), (10.0, "10x (downsampled 10x)"), (5.0, "5x (downsampled 5x)"),
                  (2.5, "2.5x (downsampled 2.5x)"), (1.0, "1x (original)")]
        for level, text in levels:
            checkbox = QCheckBox(text)
            self.level_checkboxes[level] = checkbox
            compression_layout.addWidget(checkbox)
        self.start_compression_btn = QPushButton("Start Cache Generation")
        self.start_compression_btn.clicked.connect(self.start_compression)
        self.start_compression_btn.setEnabled(False)
        compression_layout.addWidget(self.start_compression_btn)
        self.restart_compression_btn = QPushButton("Restart Cache Generation")
        self.restart_compression_btn.clicked.connect(self.restart_compression)
        self.restart_compression_btn.setEnabled(False)
        compression_layout.addWidget(self.restart_compression_btn)
        left_layout.addWidget(compression_group)
        left_layout.addStretch()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.image_title = QLabel('Select a TIFF file to view')
        self.image_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.image_title)
        
        zoom_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addStretch()
        right_layout.addLayout(zoom_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = ImageDisplayLabel()
        self.image_label.setCacheManager(self.cache_manager)
        self.image_label.cache_requested.connect(self.generate_cache_level)
        self.image_label.point_added.connect(self.on_point_added)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        
        # Right control panel
        right_control_panel = QWidget()
        right_control_panel.setMaximumWidth(300)
        right_control_layout = QVBoxLayout(right_control_panel)
        
        display_levels_group = QGroupBox("Display Level")
        display_levels_layout = QVBoxLayout(display_levels_group)
        info_label = QLabel("Available: 20x, 10x, 5x, 2.5x, 1x")
        info_label.setWordWrap(True)
        display_levels_layout.addWidget(info_label)
        level_input_layout = QHBoxLayout()
        self.level_input = QLineEdit()
        self.level_input.setPlaceholderText("Enter level (e.g., 5)")
        level_input_layout.addWidget(self.level_input)
        self.display_button = QPushButton("Display")
        self.display_button.clicked.connect(self.display_selected_level)
        level_input_layout.addWidget(self.display_button)
        display_levels_layout.addLayout(level_input_layout)
        self.cached_levels_label = QLabel("Cached levels: None")
        self.cached_levels_label.setWordWrap(True)
        display_levels_layout.addWidget(self.cached_levels_label)
        right_control_layout.addWidget(display_levels_group)
        
        community_group = QGroupBox("Communities")
        community_layout = QVBoxLayout(community_group)
        self.community_list = QListWidget()
        self.community_list.setSelectionMode(QListWidget.SingleSelection)
        community_layout.addWidget(self.community_list)
        
        # Vertical button layout
        v_btn_layout = QVBoxLayout()
        self.new_btn = QPushButton("New Community")
        self.new_btn.clicked.connect(self.on_new_community)
        v_btn_layout.addWidget(self.new_btn)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.on_ok_button)
        v_btn_layout.addWidget(self.ok_btn)
        
        self.delete_btn = QPushButton("Delete Community")
        self.delete_btn.clicked.connect(self.on_delete_community)
        v_btn_layout.addWidget(self.delete_btn)
        community_layout.addLayout(v_btn_layout)
        
        right_control_layout.addWidget(community_group)
        right_control_layout.addStretch()
        
        right_main_layout = QHBoxLayout()
        right_main_layout.addWidget(self.scroll_area, 4)
        right_main_layout.addWidget(right_control_panel, 1)
        right_layout.addLayout(right_main_layout)
        
        main_layout.addWidget(right_panel, 3)
        self.show()
    
    # ========== Show User Manual ==========
    def show_user_manual(self):
        manual_text = (
            "<h2>User Manual</h2>"
            "<p><b>Step 1:</b> Click the \"Open File\" button to open the RGB orthomosaic (TIFF image) "
            "generated by DJI Terra software from drone-captured JPG images. The TIFF files list on the "
            "left side will display all files in the folder.</p>"
            "<p><b>Step 2:</b> Under the \"Select cache levels\" section on the left, choose an image "
            "compression method. You can select from 1x, 2.5x, 5x, 10x, or 20x. If the TIFF is large and "
            "your computer is slow, choose a higher compression (e.g., 20x). If the TIFF is small or your "
            "computer is fast, choose a lower compression (e.g., 1x original). After selecting, click "
            "\"Start Cache Generation\" to compress and display the orthomosaic in the center.</p>"
            "<p><b>Step 3:</b> On the right side, click \"New Community\". An input box \"Enter community name\" "
            "will appear below the communities list. Type a name (e.g., \"example1\") and click \"OK\" to confirm.</p>"
            "<p><b>Step 4:</b> Click on the displayed image to mark multiple corner points (3 to dozens) for the "
            "community. Each clicked point will show as a red circle. When all points are set, click \"Finish\" "
            "(the OK button changes to Finish) to complete the mask annotation. The convex hull of the points "
            "will be outlined in red. Repeat steps 3 and 4 to annotate other communities. To delete a community, "
            "select it in the list and click \"Delete Community\".</p>"
            "<p><b>Step 5:</b> After all communities are annotated, click \"Save Community Mask as TXT\" at the "
            "top to save the masks as a text file.</p>"
        )
        QMessageBox.information(self, "User Manual", manual_text)
    
    # ========== Community management ==========
    def on_new_community(self):
        if self.state == "naming":
            QMessageBox.warning(self, "Warning", "Please finish naming the current community first.")
            return
        if self.state == "adding_points":
            QMessageBox.warning(self, "Warning", "Please finish adding points for the current community first.")
            return
        self.state = "naming"
        self.editing_community_index = -1
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter community name")
        self.name_edit.returnPressed.connect(self.confirm_name)
        item = QListWidgetItem()
        self.community_list.insertItem(0, item)
        self.community_list.setItemWidget(item, self.name_edit)
        self.community_list.setCurrentItem(item)
        self.name_edit.setFocus()
        self.statusBar.showMessage("Enter community name and press OK or Enter")
    
    def confirm_name(self):
        if self.state != "naming":
            return
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Community name cannot be empty.")
            return
        for comm in self.communities:
            if comm['name'] == name:
                QMessageBox.warning(self, "Warning", f"Community '{name}' already exists.")
                return
        item = self.community_list.item(0)
        self.community_list.takeItem(0)
        new_comm = {'name': name, 'points': [], 'hull': []}
        self.communities.append(new_comm)
        list_item = QListWidgetItem(name)
        self.community_list.addItem(list_item)
        self.community_list.setCurrentItem(list_item)
        self.editing_community_index = len(self.communities) - 1
        self.state = "adding_points"
        self.image_label.set_add_point_mode(True)
        self.image_label.set_fixed_points([])
        self.image_label.set_temp_line(None)
        self.statusBar.showMessage(f"Adding points for community '{name}'. Click on image to mark corners. Press OK when done.")
        self.ok_btn.setText("Finish")
        self.new_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.image_label.update_display()
    
    def on_ok_button(self):
        if self.state == "naming":
            self.confirm_name()
        elif self.state == "adding_points":
            self.finish_community_annotation()
        else:
            QMessageBox.information(self, "Info", "No action needed.")
    
    def on_point_added(self, point):
        if self.state != "adding_points" or self.editing_community_index < 0:
            return
        comm = self.communities[self.editing_community_index]
        comm['points'].append(point)
        self.image_label.set_fixed_points(comm['points'])
        self.image_label.set_temp_line(None)
        self.statusBar.showMessage(f"Point {len(comm['points'])} added. Click more or press Finish.")
    
    def finish_community_annotation(self):
        if self.editing_community_index < 0:
            return
        comm = self.communities[self.editing_community_index]
        if len(comm['points']) < 3:
            QMessageBox.warning(self, "Warning", "At least 3 points are required to form a convex hull.")
            return
        hull_float = convex_hull(comm['points'])
        if len(hull_float) < 3:
            QMessageBox.warning(self, "Warning", "Convex hull could not be formed (need at least 3 non-collinear points).")
            return
        hull_int = [QPoint(int(round(p.x())), int(round(p.y()))) for p in hull_float]
        comm['hull'] = hull_int
        self.update_polygons_from_communities()
        self.image_label.set_add_point_mode(False)
        self.image_label.set_fixed_points([])
        self.image_label.set_temp_line(None)
        self.image_label.update_display()
        self.state = "idle"
        self.ok_btn.setText("OK")
        self.new_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.statusBar.showMessage(f"Community '{comm['name']}' completed with {len(hull_int)} vertices.")
    
    def update_polygons_from_communities(self):
        polygons = []
        for comm in self.communities:
            if comm['hull']:
                polygons.append(comm['hull'])
        self.image_label.set_polygons(polygons)
    
    def on_delete_community(self):
        if self.state != "idle":
            QMessageBox.warning(self, "Warning", "Please finish current editing before deleting.")
            return
        selected_items = self.community_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a community to delete.")
            return
        item = selected_items[0]
        name = item.text()
        idx = -1
        for i, comm in enumerate(self.communities):
            if comm['name'] == name:
                idx = i
                break
        if idx < 0:
            return
        reply = QMessageBox.question(self, "Confirm Delete", f"Delete community '{name}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.communities[idx]
            self.community_list.takeItem(self.community_list.row(item))
            self.update_polygons_from_communities()
            self.image_label.update_display()
            self.statusBar.showMessage(f"Community '{name}' deleted.")
    
    # ========== Save communities to TXT ==========
    def save_communities_to_txt(self):
        if not self.geo_converter or not self.geo_converter.has_geo_info:
            QMessageBox.warning(self, "Warning", "Current TIFF has no georeferencing info. Cannot convert to lat/lon.")
            return
        valid_comms = [c for c in self.communities if c['hull']]
        if not valid_comms:
            QMessageBox.warning(self, "Warning", "No communities with valid hulls to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Community Mask", "", "Text files (*.txt)")
        if not file_path:
            return
        try:
            with open(file_path, 'w') as f:
                for comm in valid_comms:
                    name = comm['name']
                    hull_points = comm['hull']
                    latlon_pairs = []
                    for pt in hull_points:
                        latlon = self.geo_converter.pixel_to_latlon(pt.x(), pt.y())
                        if latlon is None:
                            raise Exception("Failed to convert pixel to lat/lon for one point.")
                        lat, lon = latlon
                        latlon_pairs.append((lon, lat))
                    coords_str = " ".join([f"{lon:.8f} {lat:.8f}" for lon, lat in latlon_pairs])
                    f.write(f"{name} {coords_str}\n")
            QMessageBox.information(self, "Success", f"Saved {len(valid_comms)} communities to {file_path}")
            self.statusBar.showMessage(f"Saved communities to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
    
    # ========== Cache and display methods ==========
    def zoom_in(self):
        if self.image_label:
            self.image_label.zoom_in()
    
    def zoom_out(self):
        if self.image_label:
            self.image_label.zoom_out()
    
    def display_selected_level(self):
        level_text = self.level_input.text().strip()
        if not level_text:
            QMessageBox.warning(self, "Warning", "Please enter a level to display.")
            return
        try:
            level = float(level_text)
            supported_levels = [20.0, 10.0, 5.0, 2.5, 1.0]
            if level not in supported_levels:
                QMessageBox.warning(self, "Warning", f"Unsupported level. Available: {', '.join(str(x) for x in supported_levels)}")
                return
            if not self.cache_manager.is_level_cached(level):
                QMessageBox.warning(self, "Warning", f"{level}x level not cached. Please generate it first.")
                return
            pixmap = self.cache_manager.load_level(level)
            if pixmap and not pixmap.isNull():
                self.image_label.setCurrentPixmap(pixmap, level)
                file_name = os.path.basename(self.current_file)
                self.image_title.setText(f"Image: {file_name} ({level}x)")
                self.statusBar.showMessage(f"Displayed {level}x view")
                self.image_label.setMinimumSize(pixmap.size())
                self.update_polygons_from_communities()
            else:
                QMessageBox.warning(self, "Warning", f"Failed to load {level}x image.")
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid number.")
    
    def update_cached_levels_display(self):
        if hasattr(self.cache_manager, 'file_cache_dir') and self.cache_manager.file_cache_dir:
            cached_levels = []
            for level in [20.0, 10.0, 5.0, 2.5, 1.0]:
                if self.cache_manager.is_level_cached(level):
                    cached_levels.append(f"{level}x")
            if cached_levels:
                self.cached_levels_label.setText(f"Cached levels: {', '.join(cached_levels)}")
            else:
                self.cached_levels_label.setText("Cached levels: None")
        else:
            self.cached_levels_label.setText("Cached levels: None")
    
    def load_tiff_files(self):
        self.file_list.clear()
        if not self.current_folder:
            return
        tiff_files = []
        for file in os.listdir(self.current_folder):
            if file.lower().endswith(('.tif', '.tiff')):
                tiff_files.append(os.path.join(self.current_folder, file))
        if not tiff_files:
            QMessageBox.information(self, "Info", "No TIFF files found in this folder.")
            return
        for file_path in tiff_files:
            file_name = os.path.basename(file_path)
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, file_path)
            self.file_list.addItem(item)
            if file_path == self.current_file:
                self.file_list.setCurrentItem(item)
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.current_folder = folder
            self.folder_path_label.setText(f"Current folder: {folder}")
            self.load_tiff_files()
            self.statusBar.showMessage(f"Loaded folder: {folder}")
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select TIFF file", "", "TIFF files (*.tif *.tiff)"
        )
        if file_path:
            self.current_file = file_path
            self.current_folder = os.path.dirname(file_path)
            self.folder_path_label.setText(f"Current folder: {self.current_folder}")
            self.load_tiff_files()
            self.analyze_tiff_file(file_path)
            self.statusBar.showMessage(f"Selected file: {file_path}")
    
    def on_file_selected(self, item):
        file_path = item.data(Qt.UserRole)
        self.analyze_tiff_file(file_path)
    
    def analyze_tiff_file(self, file_path):
        self.current_file = file_path
        self.image_label.setCurrentFile(file_path)
        self.image_title.setText(f"Selected: {os.path.basename(file_path)} - select cache levels")
        self.statusBar.showMessage("Analyzing TIFF file...")
        self.stop_all_loaders()
        self.image_label.reset_view()
        self.image_label.clear_polygons()
        self.communities = []
        self.community_list.clear()
        self.state = "idle"
        self.ok_btn.setText("OK")
        self.new_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.image_label.set_add_point_mode(False)
        self.image_label.set_fixed_points([])
        self.image_label.set_temp_line(None)
        self.cache_manager.set_file(file_path)
        self.level_input.clear()
        self.update_cached_levels_display()
        self.tiff_analyzer = TIFFAnalyzer(file_path)
        self.tiff_analyzer.analysis_complete.connect(self.on_analysis_complete)
        self.tiff_analyzer.error_occurred.connect(self.show_error)
        self.tiff_analyzer.start()
        self.start_compression_btn.setEnabled(True)
        self.restart_compression_btn.setEnabled(False)
    
    def on_analysis_complete(self, file_info):
        info_text = f"<b>File info:</b><br>"
        info_text += f"Size: {file_info['width']} × {file_info['height']}<br>"
        info_text += f"Bands: {file_info['bands']}<br>"
        info_text += f"Data type: {file_info['data_type']}<br>"
        info_text += f"Driver: {file_info['driver']}<br>"
        info_text += f"Pyramids: {'Yes' if file_info['has_pyramids'] else 'No'}<br>"
        if file_info.get('geo_transform') and file_info.get('projection'):
            info_text += "Geo info: Available<br>"
            self.geo_converter = GeoCoordinateConverter(file_info['geo_transform'], file_info['projection'])
        else:
            info_text += "Geo info: Not available<br>"
            self.geo_converter = None
        self.file_info_label.setText(info_text)
        self.image_label.setOriginalSize((file_info['width'], file_info['height']))
        self.statusBar.showMessage("Analysis complete. Select cache levels and generate.")
    
    def start_compression(self):
        self.selected_levels = []
        for level, checkbox in self.level_checkboxes.items():
            if checkbox.isChecked():
                self.selected_levels.append(level)
        if not self.selected_levels:
            QMessageBox.warning(self, "Warning", "Please select at least one cache level.")
            return
        self.selected_levels.sort(reverse=True)
        self.start_compression_btn.setEnabled(False)
        self.restart_compression_btn.setEnabled(True)
        for checkbox in self.level_checkboxes.values():
            checkbox.setEnabled(False)
        self.cached_levels = []
        self.generate_selected_cache_levels()
    
    def restart_compression(self):
        self.stop_all_loaders()
        self.selected_levels = []
        self.cached_levels = []
        self.level_input.clear()
        self.update_cached_levels_display()
        for checkbox in self.level_checkboxes.values():
            checkbox.setEnabled(True)
            checkbox.setChecked(False)
        self.start_compression_btn.setEnabled(True)
        self.restart_compression_btn.setEnabled(False)
        self.image_label.reset_view()
        self.image_title.setText(f"Selected: {os.path.basename(self.current_file)} - select cache levels")
        self.statusBar.showMessage("Reset. Please select cache levels.")
    
    def generate_selected_cache_levels(self):
        if not self.selected_levels:
            return
        self.progress_dialog = QProgressDialog("Generating cache...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Cache Generation")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.current_cache_index = 0
        self.generate_next_cache_level()
        self.progress_dialog.canceled.connect(self.cancel_compression)
        self.progress_dialog.show()
    
    def generate_next_cache_level(self):
        if self.current_cache_index >= len(self.selected_levels):
            self.progress_dialog.close()
            self.statusBar.showMessage("All cache levels generated.")
            self.update_cached_levels_display()
            return
        level = self.selected_levels[self.current_cache_index]
        self.statusBar.showMessage(f"Generating {level}x cache...")
        cache_generator = LevelCacheGenerator(self.current_file, self.cache_manager, level)
        cache_generator.progress_updated.connect(lambda progress, target_level: self.progress_dialog.setValue(progress))
        cache_generator.level_cached.connect(self.on_level_cached)
        cache_generator.error_occurred.connect(self.show_error)
        cache_generator.finished.connect(self.on_cache_generator_finished)
        self.cache_generators.append(cache_generator)
        cache_generator.start()
    
    def on_level_cached(self, level, pixmap):
        self.cached_levels.append(level)
        self.update_cached_levels_display()
        if level == self.selected_levels[0]:
            self.image_label.setCurrentPixmap(pixmap, level)
            file_name = os.path.basename(self.current_file)
            self.image_title.setText(f"Image: {file_name} ({level}x)")
            self.level_input.setText(str(level))
            self.image_label.setMinimumSize(pixmap.size())
    
    def on_cache_generator_finished(self):
        self.current_cache_index += 1
        self.generate_next_cache_level()
    
    def cancel_compression(self):
        self.stop_all_loaders()
        self.statusBar.showMessage("Cache generation cancelled.")
        self.restart_compression()
    
    def generate_cache_level(self, level):
        if not self.current_file:
            return
        for generator in self.cache_generators:
            if generator.target_level == level:
                return
        cache_generator = LevelCacheGenerator(self.current_file, self.cache_manager, level)
        cache_generator.level_cached.connect(self.on_level_cached)
        cache_generator.error_occurred.connect(self.show_error)
        cache_generator.finished.connect(lambda: self.cache_generators.remove(cache_generator))
        self.cache_generators.append(cache_generator)
        cache_generator.start()
    
    def stop_all_loaders(self):
        if self.tiff_analyzer and self.tiff_analyzer.isRunning():
            self.tiff_analyzer.quit()
            self.tiff_analyzer.wait()
        for generator in self.cache_generators[:]:
            if generator.isRunning():
                generator.stop()
            self.cache_generators.remove(generator)
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.image_title.setText("Failed to load image")
        self.statusBar.showMessage(f"Error: {message}")
    
    def reset_view(self):
        self.image_label.reset_view()
        self.statusBar.showMessage("View reset")
    
    def closeEvent(self, event):
        self.stop_all_loaders()
        self.cache_manager.cleanup()
        event.accept()

if __name__ == '__main__':
    if gdal_available:
        gdal.AllRegister()
    app = QApplication(sys.argv)
    font = app.font()
    font.setFamily("SimHei")
    app.setFont(font)
    viewer = TIFFViewer()
    sys.exit(app.exec_())
