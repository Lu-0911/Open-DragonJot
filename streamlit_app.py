import streamlit as st 
import cv2
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
import json
from collections import deque
from filterpy.kalman import KalmanFilter
import statistics
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import time
import psutil
import base64
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading


import streamlit as st
import threading
import psutil

# ------------------ 👥 并发访问控制逻辑 ------------------

@st.cache_resource
def get_active_sessions():
    """
    全局共享的会话计数器（跨所有用户 session 共享）。
    """
    return {"count": 0, "lock": threading.Lock()}

MAX_USERS = 1       # 同时允许的最大访问人数
MEM_THRESHOLD = 85  # 内存占用上限（百分比）

def check_user_limit():
    """
    检查是否超过访问人数或系统资源限制。
    """
    sessions = get_active_sessions()

    # 系统资源检测（防止 OOM）
    mem = psutil.virtual_memory().percent
    if mem > MEM_THRESHOLD:
        st.error(f"⚠️ 服务器资源繁忙（内存使用 {mem:.1f}%），请稍后再试。")
        st.stop()

    # 人数检测
    with sessions["lock"]:
        if sessions["count"] >= MAX_USERS:
            st.error("🚫 当前访问人数已满，请稍后再试 🙏")
            st.stop()
        else:
            sessions["count"] += 1
            st.session_state["_registered"] = True
            st.session_state["_user_id"] = id(st.session_state)

def release_user():
    """
    用户断开时释放占用的访问名额。
    """
    sessions = get_active_sessions()
    with sessions["lock"]:
        if sessions["count"] > 0:
            sessions["count"] -= 1
    print("[INFO] 当前在线用户数:", sessions["count"])

# 初始化时检测用户上限
if "_registered" not in st.session_state:
    check_user_limit()

# 用户关闭浏览器或刷新页面时自动回收名额
st.on_session_end(release_user)

# 在页面顶部显示当前状态
with st.sidebar:
    sessions = get_active_sessions()
    st.markdown(f"**👥 当前在线用户数：** {sessions['count']} / {MAX_USERS}")


# ---------------------- 路径配置 ----------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = SCRIPT_DIR / "static"
OUTPUT_DIR = SCRIPT_DIR / "temp_output"
MODELS_DIR = SCRIPT_DIR / "model"

for dir_path in [STATIC_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def clean_output_dir():
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"清理临时文件失败: {e}")

def get_model_path(relative_path):
    """将相对路径转换为绝对路径"""
    model_path = Path(relative_path)
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    return str(model_path)

import atexit
atexit.register(clean_output_dir)

# ---------------------- 通用配置 ----------------------
DRAGON_KEYPOINT_NAMES = [str(i) for i in range(1, 10)]
DRAGON_SKELETON = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
DEFAULT_NODE_COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255),
    (0, 255, 0), (255, 255, 0), (255, 128, 0),
    (255, 0, 0), (255, 0, 255), (128, 0, 255)
]
DEFAULT_LINE_COLOR = (200, 200, 200)

# ---------------------- GPU监控功能 ----------------------
class GPUMonitor:
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.gpu_name = torch.cuda.get_device_name(0) if self.use_gpu else "N/A"
        self.memory_total = self._get_total_memory()
        self.last_usage = 0
        self.last_time = time.time()
        self.fps_list = []
        
    def _get_total_memory(self):
        if self.use_gpu:
            return torch.cuda.get_device_properties(0).total_memory / (1024 **3)
        return 0
        
    def get_memory_usage(self):
        if not self.use_gpu:
            return 0, 0
        memory_used = torch.cuda.memory_allocated(0) / (1024** 3)
        memory_cached = torch.cuda.memory_reserved(0) / (1024 **3)
        return memory_used, memory_cached
        
    def update_fps(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            fps = 1 / elapsed
            self.fps_list.append(fps)
            if len(self.fps_list) > 30:
                self.fps_list.pop(0)
        self.last_time = current_time
        
    def get_average_fps(self):
        return sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0
        
    def get_cpu_usage(self):
        return psutil.cpu_percent()
        
    def get_system_memory_usage(self):
        mem = psutil.virtual_memory()
        return mem.percent

gpu_monitor = GPUMonitor()

# ---------------------- 滤波类定义 ----------------------
class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.previous_points = None
    
    def update(self, current_points):
        if self.previous_points is None:
            self.previous_points = current_points.copy()
            return current_points
        
        mask = ~np.isnan(current_points)
        smoothed = np.where(mask, 
                           self.alpha * current_points + (1 - self.alpha) * self.previous_points,
                           self.previous_points)
        self.previous_points = smoothed.copy()
        return smoothed

class KalmanFilterWrapper:
    def __init__(self, num_points, dt=0.033, process_noise=0.1, measurement_noise=5.0):
        self.filters = []
        for _ in range(num_points):
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
            kf.P = np.eye(4) * 1000
            kf.Q = np.eye(4) * process_noise
            kf.R = np.eye(2) * measurement_noise
            self.filters.append(kf)
    
    def update(self, points):
        smoothed_points = []
        for i, (x, y) in enumerate(points):
            if np.isnan(x) or np.isnan(y):
                self.filters[i].predict()
                smoothed = self.filters[i].x[:2]
            else:
                self.filters[i].predict()
                self.filters[i].update([x, y])
                smoothed = self.filters[i].x[:2]
            smoothed_points.append(smoothed)
        return np.array(smoothed_points).squeeze(-1)

# ---------------------- 平滑函数 ----------------------
def smooth_keypoints(kpts_buffer, conf_buffer=None, method="ewm", weights=None):
    if method == "none":
        if kpts_buffer:
            last_kpts = kpts_buffer[-1]
            last_conf = conf_buffer[-1] if conf_buffer else None
            return last_kpts, last_conf
        else:
            return None, None

    def to_array_with_nan(buffer, shape_expected):
        arr = []
        for k in buffer:
            if k is None:
                arr.append(np.full(shape_expected, np.nan))
            else:
                pad = [(0, shape_expected[i] - k.shape[i]) for i in range(len(shape_expected))]
                arr.append(np.pad(k, pad, mode='constant', constant_values=np.nan))
        return np.array(arr)

    num_frames = len(kpts_buffer)
    max_instances = max([0 if k is None else k.shape[0] for k in kpts_buffer])
    num_kpts = max([0 if k is None else k.shape[1] for k in kpts_buffer])
    shape_expected = (max_instances, num_kpts, 2)
    kpts_arr = to_array_with_nan(kpts_buffer, shape_expected)
    conf_arr = to_array_with_nan(conf_buffer, (max_instances, num_kpts)) if conf_buffer else None

    if method == "mean":
        smoothed_kpts = np.nanmean(kpts_arr, axis=0)
        smoothed_conf = np.nanmean(conf_arr, axis=0) if conf_buffer else None
    elif method == "weighted_mean":
        weights = np.array(weights if weights is not None else np.ones(num_frames)).reshape(-1, 1, 1, 1)
        smoothed_kpts = np.nansum(kpts_arr * weights, axis=0) / np.sum(weights * (~np.isnan(kpts_arr)), axis=0)
        smoothed_conf = np.nansum(conf_arr * weights.squeeze(-1), axis=0) / np.sum(weights.squeeze(-1) * (~np.isnan(conf_arr)), axis=0) if conf_buffer else None
    elif method == "ewm":
        alpha = 2 / (num_frames + 1)
        smoothed_kpts = kpts_arr[0].copy()
        smoothed_conf = conf_arr[0].copy() if conf_buffer else None
        for f in range(1, num_frames):
            smoothed_kpts = alpha * np.nan_to_num(kpts_arr[f]) + (1 - alpha) * smoothed_kpts
            if conf_buffer:
                smoothed_conf = alpha * np.nan_to_num(conf_arr[f]) + (1 - alpha) * smoothed_conf
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return smoothed_kpts, smoothed_conf

# ---------------------- 动作分类模型 ----------------------
class PoseCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.person_conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.person_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.person_len = 170

        self.dragon_conv1 = nn.Conv1d(3, 16, kernel_size=3, padding=1)
        self.dragon_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.dragon_len = 9

        self.fc1 = nn.Linear(64 * self.person_len + 32 * self.dragon_len, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, person_x, dragon_x):
        person_flat = person_x.reshape(person_x.size(0), 10*17, 3)
        dragon_flat = dragon_x.reshape(dragon_x.size(0), 1*9, 3)

        person_x = person_flat.transpose(1, 2)
        dragon_x = dragon_flat.transpose(1, 2)

        p = F.relu(self.person_conv1(person_x))
        p = F.relu(self.person_conv2(p))
        p = p.flatten(start_dim=1)

        d = F.relu(self.dragon_conv1(dragon_x))
        d = F.relu(self.dragon_conv2(d))
        d = d.flatten(start_dim=1)

        x = torch.cat([p, d], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_classification_model(model_path, device):
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    
    checkpoint = torch.load(model_path, map_location=device)
    model = PoseCNN(num_classes=len(checkpoint['classes']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['classes']

def build_class_inputs(person_results, dragon_results, frame_wh,
                       num_person=10, num_person_kpts=17, num_dragon_kpts=9):
    w, h = frame_wh
    person_array = np.zeros((num_person, num_person_kpts, 3), dtype=np.float32)
    dragon_array = np.zeros((1, num_dragon_kpts, 3), dtype=np.float32)

    if person_results is not None and len(person_results) > 0 and person_results[0].keypoints is not None:
        kpts_all = person_results[0].keypoints
        xy = kpts_all.xy.cpu().numpy()
        confs = kpts_all.conf.cpu().numpy()

        mean_conf = np.nanmean(confs, axis=1)
        order = np.argsort(mean_conf)[::-1]

        for i, idx in enumerate(order[:num_person]):
            k_xy = xy[idx]
            k_conf = confs[idx]
            x_norm = (k_xy[:, 0] / float(w)).astype(np.float32)
            y_norm = (k_xy[:, 1] / float(h)).astype(np.float32)
            v = k_conf.astype(np.float32)
            k_stack = np.stack([x_norm, y_norm, v], axis=1)
            L = min(num_person_kpts, k_stack.shape[0])
            person_array[i, :L, :] = k_stack[:L, :]

    if dragon_results is not None and len(dragon_results) > 0 and dragon_results[0].keypoints is not None:
        kpts_all = dragon_results[0].keypoints
        xy = kpts_all.xy.cpu().numpy()
        confs = kpts_all.conf.cpu().numpy()
        boxes = getattr(dragon_results[0], 'boxes', None)
        if boxes is not None and len(boxes) > 0 and hasattr(boxes, 'conf'):
            inst_scores = boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(inst_scores))
        else:
            mean_conf = np.nanmean(confs, axis=1)
            best_idx = int(np.argmax(mean_conf))

        k_xy = xy[best_idx]
        k_conf = confs[best_idx]
        x_norm = (k_xy[:, 0] / float(w)).astype(np.float32)
        y_norm = (k_xy[:, 1] / float(h)).astype(np.float32)
        v = k_conf.astype(np.float32)
        k_stack = np.stack([x_norm, y_norm, v], axis=1)
        L = min(num_dragon_kpts, k_stack.shape[0])
        dragon_array[0, :L, :] = k_stack[:L, :]

    return person_array, dragon_array

def classify_action(model, classes, person_array, dragon_array, device):
    if person_array.shape != (10, 17, 3):
        raise ValueError(f"person_array 形状错误，应为 (10, 17, 3)，当前 {person_array.shape}")
    if dragon_array.shape != (1, 9, 3):
        raise ValueError(f"dragon_array 形状错误，应为 (1, 9, 3)，当前 {dragon_array.shape}")

    classes_dict = {"BZ": "八字类", "DC":"单侧类", "CT": "穿腾类", "FG": "翻滚类", "YL": "游龙类", "ZX": "造型类"}
    classes = [classes_dict.get(c, c) for c in classes]

    person_tensor = torch.tensor(person_array, dtype=torch.float32).unsqueeze(0).to(device)
    dragon_tensor = torch.tensor(dragon_array, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(person_tensor, dragon_tensor)
        probs = F.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return classes[pred]

def put_chinese_text(img, text, pos=(30,80), color=(0,255,0), font_size=32):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("simsun.ttc", font_size)
        except:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=color)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img


# ---------------------- 图像检测 ----------------------
def process_image(person_model, dragon_model, img_path, confs, device,
                  save_json=False, save_txt=False, 
                  single_dragon=True, only_person=False, only_dragon=False,
                  classify=False, classify_model=None, verbose=False,
                  node_colors=None, node_size=10, line_color=None, line_thickness=6):
    node_colors = node_colors or DEFAULT_NODE_COLORS
    line_color = line_color or DEFAULT_LINE_COLOR
    
    person_conf, dragon_conf, person_kpt_conf, dragon_kpt_conf = confs

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    
    person_results = None
    dragon_results = None
    

    person_model_path = Path(person_model)
    if not person_model_path.is_absolute():
        person_model_path = MODELS_DIR / person_model_path
    
    dragon_model_path = Path(dragon_model)
    if not dragon_model_path.is_absolute():
        dragon_model_path = MODELS_DIR / dragon_model_path

    if not only_dragon:
        model_person = YOLO(str(person_model_path))
        model_person.to(device)
        person_results = model_person(img, conf=person_conf, verbose=verbose)

    if not only_person:
        model_dragon = YOLO(str(dragon_model_path))
        model_dragon.to(device)
        dragon_results = model_dragon(img, conf=dragon_conf, verbose=verbose)
    
    if only_dragon or only_person:
        classify = False

    img_out = img.copy()
    if person_results is not None:
        img_out = person_results[0].plot(boxes=False)

    if dragon_results and dragon_results[0].keypoints is not None:
        boxes = dragon_results[0].boxes

        if boxes is not None and len(boxes) > 0:
            kpts = dragon_results[0].keypoints.xy.cpu().numpy()
            conf = dragon_results[0].keypoints.conf.cpu().numpy()
            if single_dragon:
                best_idx = np.argmax(boxes.conf.cpu().numpy())
                kpts = kpts[best_idx:best_idx+1]
                conf = conf[best_idx:best_idx+1]

            for i, kp_set in enumerate(kpts):
                for j, ((x, y), c) in enumerate(zip(kp_set, conf[i])):
                    if c > dragon_kpt_conf:
                        color = node_colors[j % len(node_colors)]
                        cv2.circle(img_out, (int(x), int(y)), node_size, color, -1, lineType=cv2.LINE_AA)
                        cv2.putText(img_out, str(j + 1), (int(x) + 5, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                for a, b in DRAGON_SKELETON:
                    if conf[i][a] > dragon_kpt_conf and conf[i][b] > dragon_kpt_conf:
                        pt1, pt2 = tuple(map(int, kp_set[a])), tuple(map(int, kp_set[b]))
                        cv2.line(img_out, pt1, pt2, line_color, line_thickness)

    if classify and classify_model is not None:
        classify_model_obj, classes = load_classification_model(MODELS_DIR / classify_model, device)
        num_person, num_person_kpts = 10, 17
        num_dragon_kpts = 9
        person_array = np.zeros((num_person, num_person_kpts, 3), dtype=np.float32)
        dragon_array = np.zeros((1, num_dragon_kpts, 3), dtype=np.float32)

        if person_results is not None and len(person_results[0].keypoints) > 0:
            kpts_all = person_results[0].keypoints
            confs = kpts_all.conf.cpu().numpy()
            xy = kpts_all.xy.cpu().numpy()
            mean_conf = confs.mean(axis=1)
            order = np.argsort(mean_conf)[::-1]
            for i, idx in enumerate(order[:num_person]):
                x = xy[idx, :, 0]
                y = xy[idx, :, 1]
                v = confs[idx]
                person_array[i, :len(x), :] = np.stack([x, y, v], axis=1)

        if dragon_results is not None and len(dragon_results[0].keypoints) > 0:
            kpts = dragon_results[0].keypoints.xy.cpu().numpy()[0]
            conf = dragon_results[0].keypoints.conf.cpu().numpy()[0]
            dragon_array[0, :len(kpts), :] = np.stack([kpts[:, 0], kpts[:, 1], conf], axis=1)

        action_label = classify_action(classify_model_obj, classes, person_array, dragon_array, device)
        img_out = put_chinese_text(img_out, f"Action: {action_label}")

    output_img_path = OUTPUT_DIR / "output_image.jpg"
    cv2.imwrite(str(output_img_path), img_out)

    if save_json:
        if person_results is not None:
            person_labels = json.loads(person_results[0].to_json())
            (OUTPUT_DIR / "person.json").write_text(json.dumps(person_labels, indent=2), encoding='utf-8')
        if dragon_results is not None:
            dragon_labels = json.loads(dragon_results[0].to_json())
            (OUTPUT_DIR / "dragon.json").write_text(json.dumps(dragon_labels, indent=2), encoding='utf-8')

    if save_txt:
        if person_results is not None:
            person_results[0].save_txt(str(OUTPUT_DIR / "person.txt"))
        if dragon_results is not None:
            dragon_results[0].save_txt(str(OUTPUT_DIR / "dragon.txt"))

    return img_out, output_img_path

# ---------------------- 视频检测 ----------------------
def process_video(person_model, dragon_model, video_path, confs, realtime_filter_method, smooth, device,
                  save_json=False, save_txt=False, show_preview=True,
                  single_dragon=True, only_person=False, only_dragon=False, 
                  classify=False, classify_model=None, save_video=False, verbose=False,
                  node_colors=None, node_size=10, line_color=None, line_thickness=6,
                  gpu_monitor=None, status_callback=None):
    node_colors = node_colors or DEFAULT_NODE_COLORS
    line_color = line_color or DEFAULT_LINE_COLOR
    
    person_conf, dragon_conf, person_kpt_conf, dragon_kpt_conf = confs

    person_model_path = Path(person_model)
    if not person_model_path.is_absolute():
        person_model_path = MODELS_DIR / person_model_path
    
    dragon_model_path = Path(dragon_model)
    if not dragon_model_path.is_absolute():
        dragon_model_path = MODELS_DIR / dragon_model_path

    model_person = YOLO(str(person_model_path)).to(device) if not only_dragon else None
    model_dragon = YOLO(str(dragon_model_path)).to(device) if not only_person else None

    realtime_filter = None
    if realtime_filter_method == 'ema':
        realtime_filter = EMAFilter(alpha=0.3)
    elif realtime_filter_method == 'kalman':
        realtime_filter = KalmanFilterWrapper(
            num_points=len(DRAGON_KEYPOINT_NAMES),
            dt=1/30.0,
            process_noise=0.2,
            measurement_noise=10.0
        )

    smooth_window = 3
    frame_id = 0
    dragon_kpts_buffer = deque(maxlen=2 * smooth_window + 1)
    dragon_conf_buffer = deque(maxlen=2 * smooth_window + 1)
    frame_buffer = deque(maxlen=2 * smooth_window + 1)
    
    current_smoothed_kpts = None
    current_smoothed_conf = None

    if only_dragon or only_person:
        classify = False

    classify_model_obj = None
    classes = None
    if classify and classify_model is not None:
        classify_model_obj, classes = load_classification_model(MODELS_DIR / classify_model, device)
        class_buffer = deque(maxlen=30)
        display_class = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video_path = None
    
    if save_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 确保FPS有效
        if fps <= 0:
            fps = 30  # 使用默认FPS
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = OUTPUT_DIR / "output_video.mp4"
        try:
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
            if not out.isOpened():
                raise Exception(f"无法初始化视频写入器，可能是编解码器问题: {output_video_path}")
        except Exception as e:
            st.error(f"视频写入初始化失败: {str(e)}")
            return None

    # 预览占位符
    preview_placeholder = st.empty() if show_preview else None
    gpu_status_placeholder = st.empty() if gpu_monitor and gpu_monitor.use_gpu else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 更新进度
            if status_callback:
                progress = min(frame_id / total_frames, 1.0)
                status_callback(f"处理进度: {frame_id}/{total_frames} 帧", progress)

            # 更新GPU监控
            if gpu_monitor and gpu_status_placeholder:
                gpu_monitor.update_fps()
                mem_used, mem_cached = gpu_monitor.get_memory_usage()
                cpu_usage = gpu_monitor.get_cpu_usage()
                sys_mem_usage = gpu_monitor.get_system_memory_usage()
                avg_fps = gpu_monitor.get_average_fps()
                
                gpu_status = f"""
                **GPU状态监控**  
                GPU: {gpu_monitor.gpu_name}  
                显存使用: {mem_used:.2f}GB / {gpu_monitor.memory_total:.2f}GB  
                缓存显存: {mem_cached:.2f}GB  
                平均帧率: {avg_fps:.1f} FPS  
                CPU使用率: {cpu_usage}%  
                系统内存使用率: {sys_mem_usage}%
                """
                gpu_status_placeholder.markdown(gpu_status)

            frame_buffer.append(frame.copy())
            person_results = model_person(frame, conf=person_conf, verbose=verbose) if model_person else None
            dragon_results = model_dragon(frame, conf=dragon_conf, verbose=verbose) if model_dragon else None

            if dragon_results and dragon_results[0].keypoints is not None: 
                boxes = dragon_results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    kpts = dragon_results[0].keypoints.xy.cpu().numpy()
                    conf = dragon_results[0].keypoints.conf.cpu().numpy()
                    if single_dragon:
                        best_idx = np.argmax(boxes.conf.cpu().numpy())
                        kpts = kpts[best_idx:best_idx+1]
                        conf = conf[best_idx:best_idx+1]
                    
                    if realtime_filter is not None and len(kpts) > 0:
                        current_kpts = kpts[0]
                        smoothed_kpts = realtime_filter.update(current_kpts)
                        current_smoothed_kpts = smoothed_kpts.reshape(1, -1, 2)
                        current_smoothed_conf = conf
                    else:
                        current_smoothed_kpts = kpts
                        current_smoothed_conf = conf
                    
                    dragon_kpts_buffer.append(current_smoothed_kpts)
                    dragon_conf_buffer.append(current_smoothed_conf)

            else:
                empty_kpts = np.full((1, len(DRAGON_KEYPOINT_NAMES), 2), np.nan)
                empty_conf = np.full((1, len(DRAGON_KEYPOINT_NAMES)), np.nan)
                dragon_kpts_buffer.append(empty_kpts)
                dragon_conf_buffer.append(empty_conf)
                current_smoothed_kpts = None
                current_smoothed_conf = None

            img = frame.copy()
            if person_results is not None:
                img = person_results[0].plot(boxes=False)

            draw_kpts = None
            draw_conf = None

            if len(dragon_kpts_buffer) == dragon_kpts_buffer.maxlen and smooth != "none":
                smoothed_kpts, smoothed_conf = smooth_keypoints(dragon_kpts_buffer, dragon_conf_buffer, method=smooth)
                if smoothed_kpts is not None and len(smoothed_kpts) > 0:
                    draw_kpts = smoothed_kpts
                    draw_conf = smoothed_conf
            elif current_smoothed_kpts is not None and len(current_smoothed_kpts) > 0:
                draw_kpts = current_smoothed_kpts
                draw_conf = current_smoothed_conf
            elif dragon_kpts_buffer:
                draw_kpts = dragon_kpts_buffer[-1]
                draw_conf = dragon_conf_buffer[-1] if dragon_conf_buffer else None

            if draw_kpts is not None and len(draw_kpts) > 0:
                kp_set, conf_set = draw_kpts[0], draw_conf[0]
                for j, ((x, y), c) in enumerate(zip(kp_set, conf_set)):
                    if not np.isnan(x) and not np.isnan(y) and c > dragon_kpt_conf:
                        color = node_colors[j % len(node_colors)]
                        cv2.circle(img, (int(x), int(y)), node_size, color, -1)
                        cv2.putText(img, str(j + 1), (int(x)+5, int(y)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                for a, b in DRAGON_SKELETON:
                    if (conf_set[a] > dragon_kpt_conf and conf_set[b] > dragon_kpt_conf and
                        not np.isnan(kp_set[a][0]) and not np.isnan(kp_set[b][0])):
                        cv2.line(img, tuple(map(int, kp_set[a])), tuple(map(int, kp_set[b])), line_color, line_thickness)

                if classify and classes is not None and draw_kpts is not None:
                    try:
                        frame_wh = (w, h)
                        person_array, dragon_array = build_class_inputs(person_results, dragon_results, frame_wh)
                        label = classify_action(classify_model_obj, classes, person_array, dragon_array, device)
                    except Exception as e:
                        label = None

                    if label is not None:
                        class_buffer.append(label)

                    if len(class_buffer) == 30:
                        stable_class = statistics.mode(class_buffer)
                        display_class = stable_class

            if classify and display_class is not None:
                img = put_chinese_text(img, f"Action: {display_class}")

            # 实时预览
            if show_preview and preview_placeholder is not None:
                preview_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(preview_img, caption=f"帧 {frame_id}", width='stretch')

            if save_video and 'out' in locals() and out.isOpened():
                try:
                    out.write(img)
                except Exception as e:
                    st.error(f"写入视频帧失败 (帧 {frame_id}): {str(e)}")
                    st.error(f"帧尺寸: {img.shape}, 视频尺寸: ({w}, {h})")

            frame_id += 1

    except Exception as e:
        st.error(f"视频处理过程中出错: {str(e)}")
    finally:
        if save_video and 'out' in locals():
            # 确保所有帧都被写入
            for i in range(smooth_window):
                if i < len(frame_buffer):
                    try:
                        out.write(frame_buffer[i])
                    except:
                        pass
            out.release()
            # 确保文件正确关闭
            cv2.destroyAllWindows()
            
            # 验证视频文件
            if output_video_path.exists():
                cap_check = cv2.VideoCapture(str(output_video_path))
                if not cap_check.isOpened():
                    st.error(f"生成的视频文件无法打开，可能已损坏: {output_video_path}")
                else:
                    check_ret, _ = cap_check.read()
                    if not check_ret:
                        st.error(f"生成的视频文件为空或损坏，无法读取帧: {output_video_path}")
                    cap_check.release()
            else:
                st.error(f"视频文件未生成: {output_video_path}")

        cap.release()
        if gpu_status_placeholder:
            gpu_status_placeholder.empty()
            
    return output_video_path

# ---------------------- 摄像头检测 ----------------------
def process_camera(person_model, dragon_model, cam_id, confs, realtime_filter_method, device,
                   single_dragon=True, only_person=False, only_dragon=False,
                   classify=False, classify_model=None, save_video=False,
                   preview_placeholder=None, stop_flag=None,
                   node_colors=None, node_size=10, line_color=None, line_thickness=6,
                   gpu_monitor=None):
    node_colors = node_colors or DEFAULT_NODE_COLORS
    line_color = line_color or DEFAULT_LINE_COLOR
    
    person_conf, dragon_conf, person_kpt_conf, dragon_kpt_conf = confs
    
    person_model_path = Path(person_model)
    if not person_model_path.is_absolute():
        person_model_path = MODELS_DIR / person_model_path
    
    dragon_model_path = Path(dragon_model)
    if not dragon_model_path.is_absolute():
        dragon_model_path = MODELS_DIR / dragon_model_path

    model_person = YOLO(str(person_model_path)).to(device) if not only_dragon else None
    model_dragon = YOLO(str(dragon_model_path)).to(device) if not only_person else None

    realtime_filter = None
    if realtime_filter_method == 'ema':
        realtime_filter = EMAFilter(alpha=0.4)
    elif realtime_filter_method == 'kalman':
        realtime_filter = KalmanFilterWrapper(
            num_points=len(DRAGON_KEYPOINT_NAMES),
            dt=1/30.0,
            process_noise=0.1,
            measurement_noise=5.0
        )

    if only_dragon or only_person:
        classify = False

    classify_model_obj = None
    classes = None
    if classify and classify_model is not None:
        classify_model_obj, classes = load_classification_model(MODELS_DIR / classify_model, device)
        class_buffer = deque(maxlen=30)
        display_class = None

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        preview_placeholder.error(f"无法打开摄像头 {cam_id}，请检查摄像头连接")
        return
    
    # 视频录制设置
    video_writer = None
    output_video_path = None
    if save_video:
        output_video_path = OUTPUT_DIR / "camera_output.mp4"
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20  # 使用默认FPS
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
            if not video_writer.isOpened():
                raise Exception(f"无法初始化摄像头视频写入器，可能是编解码器问题")
        except Exception as e:
            st.error(f"摄像头视频写入初始化失败: {str(e)}")
            save_video = False

    # 状态显示
    status_text = st.empty()
    status_text.info("摄像头运行中...")
    
    # GPU监控占位符
    gpu_status_placeholder = st.empty() if gpu_monitor and gpu_monitor.use_gpu else None

    try:
        while True:
            # 检查停止条件
            if stop_flag and stop_flag():
                break
                
            # 更新GPU监控
            if gpu_monitor and gpu_status_placeholder:
                gpu_monitor.update_fps()
                mem_used, mem_cached = gpu_monitor.get_memory_usage()
                cpu_usage = gpu_monitor.get_cpu_usage()
                sys_mem_usage = gpu_monitor.get_system_memory_usage()
                avg_fps = gpu_monitor.get_average_fps()
                
                gpu_status = f"""
                **GPU状态监控**  
                GPU: {gpu_monitor.gpu_name}  
                显存使用: {mem_used:.2f}GB / {gpu_monitor.memory_total:.2f}GB  
                缓存显存: {mem_cached:.2f}GB  
                平均帧率: {avg_fps:.1f} FPS  
                CPU使用率: {cpu_usage}%  
                系统内存使用率: {sys_mem_usage}%
                """
                gpu_status_placeholder.markdown(gpu_status)

            ret, frame = cap.read()
            if not ret:
                preview_placeholder.error("无法从摄像头读取帧")
                break
                
            person_results = model_person(frame, conf=person_conf, verbose=False) if model_person else None
            dragon_results = model_dragon(frame, conf=dragon_conf, verbose=False) if model_dragon else None

            img = frame.copy()
            if person_results is not None:
                img = person_results[0].plot(boxes=False)

            if dragon_results and dragon_results[0].keypoints is not None:
                boxes = dragon_results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    kpts = dragon_results[0].keypoints.xy.cpu().numpy()
                    conf = dragon_results[0].keypoints.conf.cpu().numpy()
                    if single_dragon:
                        best_idx = np.argmax(boxes.conf.cpu().numpy())
                        kpts = kpts[best_idx:best_idx+1]
                        conf = conf[best_idx:best_idx+1]
                    
                    if realtime_filter is not None and len(kpts) > 0:
                        current_kpts = kpts[0]
                        smoothed_kpts = realtime_filter.update(current_kpts)
                        kpts[0] = smoothed_kpts.reshape(1, -1, 2)
                    
                    for i, kp_set in enumerate(kpts):
                        for j, ((x, y), c) in enumerate(zip(kp_set, conf[i])):
                            if c > dragon_kpt_conf:
                                color = node_colors[j % len(node_colors)]
                                cv2.circle(img, (int(x), int(y)), node_size, color, -1)
                                cv2.putText(img, str(j + 1), (int(x)+5, int(y)-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        for (a, b) in DRAGON_SKELETON:
                            if conf[i][a] > dragon_kpt_conf and conf[i][b] > dragon_kpt_conf:
                                pt1, pt2 = tuple(map(int, kp_set[a])), tuple(map(int, kp_set[b]))
                                cv2.line(img, pt1, pt2, line_color, line_thickness)
                            
                    if classify and classes is not None:
                        try:
                            frame_wh = (frame.shape[1], frame.shape[0])
                            person_array, dragon_array = build_class_inputs(person_results, dragon_results, frame_wh)
                            label = classify_action(classify_model_obj, classes, person_array, dragon_array, device)
                        except Exception as e:
                            label = None

                        if label is not None:
                            class_buffer.append(label)

                        if len(class_buffer) == 30:
                            stable_class = statistics.mode(class_buffer)
                            display_class = stable_class
            
            if classify and display_class is not None:
                img = put_chinese_text(img, f"Action: {display_class}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(img_rgb, channels="RGB", width='stretch')

            if save_video and video_writer is not None and video_writer.isOpened():
                try:
                    video_writer.write(img)
                except Exception as e:
                    st.error(f"写入摄像头帧失败: {str(e)}")
                    st.error(f"帧尺寸: {img.shape}, 视频尺寸: ({frame_width}, {frame_height})")

            # 小延迟防止UI卡顿
            time.sleep(0.01)

    except Exception as e:
        preview_placeholder.error(f"摄像头处理出错: {str(e)}")
    
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            cv2.destroyAllWindows()
            
            # 验证视频文件
            if output_video_path and output_video_path.exists():
                cap_check = cv2.VideoCapture(str(output_video_path))
                if not cap_check.isOpened():
                    st.error(f"生成的摄像头视频文件无法打开，可能已损坏: {output_video_path}")
                else:
                    check_ret, _ = cap_check.read()
                    if not check_ret:
                        st.error(f"生成的摄像头视频文件为空或损坏，无法读取帧: {output_video_path}")
                    cap_check.release()
            elif save_video:
                st.error(f"摄像头视频文件未生成: {output_video_path}")
        
        status_text.success("摄像头已停止")
        if gpu_status_placeholder:
            gpu_status_placeholder.empty()
            
        return output_video_path
    

def process_camera_stream(params, gpu_monitor=None):
    """完整摄像头识别逻辑"""
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            # --- 初始化参数 ---
            self.model_person = None
            self.model_dragon = None
            self.device = params['device']
            self.node_colors = params['node_colors']
            self.line_color = params['line_color']
            self.node_size = params['node_size']
            self.line_thickness = params['line_thickness']
            self.person_conf, self.dragon_conf, self.person_kpt_conf, self.dragon_kpt_conf = params['confs']
            self.single_dragon = params.get('single_dragon', True)
            self.only_person = params.get('only_person', False)
            self.only_dragon = params.get('only_dragon', False)

            # --- 分类模型 ---
            self.classify = params.get('classify', False)
            self.classify_model = params.get('classify_model', None)
            self.classify_model_obj = None
            self.classes = None
            self.class_buffer = deque(maxlen=30)
            self.display_class = None

            # --- 实时平滑滤波器 ---
            self.realtime_filter_method = params.get('realtime_filter_method', None)
            if self.realtime_filter_method == 'ema':
                self.realtime_filter = EMAFilter(alpha=0.4)
            elif self.realtime_filter_method == 'kalman':
                self.realtime_filter = KalmanFilterWrapper(
                    num_points=len(DRAGON_KEYPOINT_NAMES),
                    dt=1 / 30.0,
                    process_noise=0.1,
                    measurement_noise=5.0,
                )
            else:
                self.realtime_filter = None

            # --- 视频保存 ---
            self.save_video = params.get('save_video', False)
            self.output_video_path = OUTPUT_DIR / "camera_output.mp4"
            self.video_writer = None
            self.fps = 0.0
            self.last_time = time.time()

        def _init_models(self):
            """懒加载模型"""
            if self.model_person is None and not self.only_dragon:
                person_model_path = Path(params['person_model'])
                if not person_model_path.is_absolute():
                    person_model_path = MODELS_DIR / person_model_path
                self.model_person = YOLO(str(person_model_path)).to(self.device)

            if self.model_dragon is None and not self.only_person:
                dragon_model_path = Path(params['dragon_model'])
                if not dragon_model_path.is_absolute():
                    dragon_model_path = MODELS_DIR / dragon_model_path
                self.model_dragon = YOLO(str(dragon_model_path)).to(self.device)

            if self.classify and self.classify_model_obj is None and self.classify_model:
                self.classify_model_obj, self.classes = load_classification_model(
                    MODELS_DIR / self.classify_model, self.device
                )

            print("[INFO] 模型加载完成，可开始实时检测")

        def _init_video_writer(self, frame_shape):
            if not self.save_video:
                return
            h, w = frame_shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(self.output_video_path), fourcc, 20.0, (w, h))

        def recv(self, frame):
            # --- 模型初始化 ---
            self._init_models()
            img = frame.to_ndarray(format="bgr24")

            # --- FPS 计算 ---
            now = time.time()
            dt = now - self.last_time
            if dt > 0:
                self.fps = 1.0 / dt
            self.last_time = now

            # --- 人体检测 ---
            results_person = None
            if self.model_person:
                results_person = self.model_person(img, conf=self.person_conf, verbose=False)
                img = results_person[0].plot(boxes=False)

            # --- 龙检测 ---
            results_dragon = None
            if self.model_dragon:
                results_dragon = self.model_dragon(img, conf=self.dragon_conf, verbose=False)

            # --- 龙关键点绘制 ---
            if results_dragon and results_dragon[0].keypoints is not None:
                boxes = results_dragon[0].boxes
                kpts = results_dragon[0].keypoints.xy.cpu().numpy()
                conf = results_dragon[0].keypoints.conf.cpu().numpy()

                if self.single_dragon and boxes is not None and len(boxes) > 0:
                    best_idx = np.argmax(boxes.conf.cpu().numpy())
                    kpts = kpts[best_idx:best_idx + 1]
                    conf = conf[best_idx:best_idx + 1]

                if self.realtime_filter is not None and len(kpts) > 0:
                    smoothed = self.realtime_filter.update(kpts[0])
                    kpts[0] = smoothed.reshape(-1, 2)

                for i, kp_set in enumerate(kpts):
                    for j, ((x, y), c) in enumerate(zip(kp_set, conf[i])):
                        if c > self.dragon_kpt_conf:
                            color = self.node_colors[j % len(self.node_colors)]
                            cv2.circle(img, (int(x), int(y)), self.node_size, color, -1)
                            cv2.putText(img, str(j + 1), (int(x)+5, int(y)-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    for (a, b) in DRAGON_SKELETON:
                        if conf[i][a] > self.dragon_kpt_conf and conf[i][b] > self.dragon_kpt_conf:
                            pt1, pt2 = tuple(map(int, kp_set[a])), tuple(map(int, kp_set[b]))
                            cv2.line(img, pt1, pt2, self.line_color, self.line_thickness)

            # --- 动作分类 ---
            if self.classify and self.classify_model_obj is not None:
                try:
                    frame_wh = (img.shape[1], img.shape[0])
                    person_array, dragon_array = build_class_inputs(results_person, results_dragon, frame_wh)
                    label = classify_action(self.classify_model_obj, self.classes, person_array, dragon_array, self.device)
                    if label is not None:
                        self.class_buffer.append(label)
                        if len(self.class_buffer) == self.class_buffer.maxlen:
                            stable_class = statistics.mode(self.class_buffer)
                            if stable_class != self.display_class:
                                self.display_class = stable_class
                except Exception:
                    pass

                if self.display_class:
                    img = put_chinese_text(
                        img, 
                        f"Action:{self.display_class}",
                        pos=(30, 60),
                        color=(0, 255, 0),
                        font_size=32
                    )

            # --- 视频保存 ---
            if self.save_video and self.video_writer is None:
                self._init_video_writer(img.shape)
            if self.video_writer:
                try:
                    self.video_writer.write(img)
                except Exception as e:
                    print(f"[WARN] 写入视频失败: {e}")

            # --- GPU状态监控（可选）---
            if gpu_monitor:
                gpu_monitor.update_fps()
                if gpu_monitor.use_gpu:
                    mem_used, mem_cached = gpu_monitor.get_memory_usage()
                    print(f"[GPU] FPS {self.fps:.1f}, Mem {mem_used:.2f} GB")
                else:
                    print(f"[CPU] FPS {self.fps:.1f}")

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def on_stop(self):
            if self.video_writer:
                self.video_writer.release()
                print("[INFO] 录制视频已保存:", self.output_video_path)

    # --- 启动 WebRTC ---
    webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )



     
# ---------------------- 检测可用摄像头 ----------------------
@st.cache_resource(show_spinner=False)
def get_available_cameras(max_test=5):
    available_cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras


# ---------------------- Streamlit Web界面 ----------------------

def main():
    st.set_page_config(
        page_title="Open DragonJot - 舞龙动作识别检测系统",
        page_icon="🐉",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 初始化会话状态
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "功能演示"
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'stop_flag' not in st.session_state:
        st.session_state.stop_flag = False
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None

    # 侧边栏 - 顶部Logo和文字
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 0;">
    """, unsafe_allow_html=True)
    
    try:
        st.sidebar.image(str(STATIC_DIR / "logo.jpg"), width='stretch')
    except Exception as e:
        st.sidebar.warning(f"未找到Logo文件: {STATIC_DIR / 'logo.jpg'}")
    
    # 添加Logo下方的文字信息
    st.sidebar.markdown("""
    <h3 style="text-align: center; margin-top: 10px; margin-bottom: 5px;">Open DragonJot</h3>
    <p style="text-align: center; margin-top: 0; color: #666;">点睛AI开源版</p>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏 - 选项卡选择
    st.sidebar.title("导航菜单")
    # 为每个按钮添加唯一的key参数
    if st.sidebar.button("功能演示", width='stretch', 
                         type="primary" if st.session_state.current_tab == "功能演示" else "secondary",
                         key="btn_feature_demo"):
        st.session_state.current_tab = "功能演示"
    
    if st.sidebar.button("快速体验", width='stretch', 
                         type="primary" if st.session_state.current_tab == "快速体验" else "secondary",
                         key="btn_quick_start"):
        st.session_state.current_tab = "快速体验"
    
    # 新增参数设置独立板块
    if st.sidebar.button("参数设置", width='stretch', 
                         type="primary" if st.session_state.current_tab == "参数设置" else "secondary",
                         key="btn_settings"):
        st.session_state.current_tab = "参数设置"
    
    # 新增开发团队板块
    if st.sidebar.button("开发团队", width='stretch', 
                         type="primary" if st.session_state.current_tab == "开发团队" else "secondary",
                         key="btn_team"):
        st.session_state.current_tab = "开发团队"

    # 功能演示页面
    if st.session_state.current_tab == "功能演示":
        st.markdown("<h1 style='text-align: center;'>Open DragonJot - 功能演示</h1>", unsafe_allow_html=True)
        
        # 展示静态文件夹中的GIF
        st.subheader("  1. 人龙一体姿态识别")
        try:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # 读取 GIF 文件并编码
                gif_path = STATIC_DIR / "skeleton_demo.gif"
                if os.path.exists(gif_path):
                    with open(gif_path, "rb") as f:
                        gif_data = f.read()
                        gif_base64 = base64.b64encode(gif_data).decode()
                    
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: center; flex-direction: column; align-items: center;">
                            <img src="data:image/gif;base64,{gif_base64}" style="width: 450px; max-width: 100%;">
                            <p style="text-align: center; font-size: 14px; color: #6B7280;">朱俊鹏教练在朱家角杯开场表演</p>
                            <p style="text-align: center; font-size: 14px; color: #6B7280;">龙骨架关键点实时识别演示</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(f"未找到演示文件: {gif_path}")
        except Exception as e:
            st.warning(f"加载演示文件出错: {e}")

        st.subheader("  2. 五大动作分类")
        try:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # 读取 GIF 文件并编码
                gif_path = STATIC_DIR / "classification_demo.gif"
                if os.path.exists(gif_path):
                    with open(gif_path, "rb") as f:
                        gif_data = f.read()
                        gif_base64 = base64.b64encode(gif_data).decode()
                    
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: center; flex-direction: column; align-items: center;">
                            <img src="data:image/gif;base64,{gif_base64}" style="width: 450px; max-width: 100%;">
                            <p style="text-align: center; font-size: 14px; color: #6B7280;">复旦龙狮协会日常训练</p>
                            <p style="text-align: center; font-size: 14px; color: #6B7280;">舞龙动作自动分类演示</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(f"未找到演示文件: {gif_path}")
        except Exception as e:
            st.warning(f"加载演示文件出错: {e}")

        # st.subheader("  3. 动作实时评分")
        # try:
        #     col1, col2, col3 = st.columns([1, 2, 1])
        #     with col2:
        #         # 读取 GIF 文件并编码
        #         gif_path = STATIC_DIR / "score_demo.gif"
        #         if os.path.exists(gif_path):
        #             with open(gif_path, "rb") as f:
        #                 gif_data = f.read()
        #                 gif_base64 = base64.b64encode(gif_data).decode()
                    
        #             st.markdown(
        #                 f"""
        #                 <div style="display: flex; justify-content: center; flex-direction: column; align-items: center;">
        #                     <img src="data:image/gif;base64,{gif_base64}" style="width: 450px; max-width: 100%;">
        #                     <p style="text-align: center; font-size: 14px; color: #6B7280;">点睛AI专业版</p>
        #                     <p style="text-align: center; margin-top: 10px; font-size: 14px; color: #6B7280;">舞龙动作AI打分演示</p>
        #                 </div>
        #                 """, 
        #                 unsafe_allow_html=True
        #             )
        #         else:
        #             st.warning(f"未找到演示文件: {gif_path}")
        # except Exception as e:
        #     st.warning(f"加载演示文件出错: {e}")
        
        st.markdown("""
        ### 功能说明
        - 系统可实时识别龙骨架的9个关键节点
        - 支持五大类舞龙动作的自动分类
        - 支持图像、视频、摄像头输入
        - 提供节点颜色、大小等个性化设置
        - 可使用GPU加速提高处理效率
        
        点击左侧"快速体验"开始使用系统功能。
        """)

        st.markdown("""
        `开源版本暂未开放动作评分功能，有兴趣可访问点睛AI专业版`
        `咨询热线：15398180360`
        """)

    # 快速体验页面
    elif st.session_state.current_tab == "快速体验":
        st.markdown("<h1 style='text-align: center;'>Open DragonJot</h1>", unsafe_allow_html=True)
        
        # 横向模式选择
        mode = st.radio(
            "选择检测模式:",
            ["📷 图片模式", "🎥 视频模式", "📹 摄像头模式"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # 简化模式映射
        mode_mapping = {
            "📷 图片模式": "image",
            "🎥 视频模式": "video", 
            "📹 摄像头模式": "camera"
        }
        current_mode = mode_mapping[mode]
        
        # 主内容区域 - 图片模式
        if current_mode == "image":
            uploaded_file = st.file_uploader("选择图片文件", type=['jpg', 'jpeg', 'png'], key="upload_image")
            
            if uploaded_file is not None:
                # 显示预览和结果在同一区域
                preview_placeholder = st.empty()
                
                # 临时保存文件并显示原图预览
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    img_path = tmp_file.name
                
                # 显示原图预览
                original_img = cv2.imread(img_path)
                preview_placeholder.image(
                    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 
                    caption="原图预览",
                    width='stretch'
                )
                
                if st.button("开始检测图片", width='stretch', type="primary", key="btn_process_image"):
                    with st.spinner("正在处理图片..."):
                        # 获取参数设置
                        params = get_params()
                        
                        # 显示GPU状态（如果使用GPU）
                        gpu_status_placeholder = None
                        if params['device'].type == 'cuda':
                            gpu_status_placeholder = st.empty()
                            mem_used, mem_cached = gpu_monitor.get_memory_usage()
                            gpu_status = f"""
                            **GPU状态**  
                            显存使用: {mem_used:.2f}GB / {gpu_monitor.memory_total:.2f}GB  
                            """
                            gpu_status_placeholder.markdown(gpu_status)
                        
                        result_img, output_img_path = process_image(
                            params['person_model'], params['dragon_model'], img_path, params['confs'], params['device'],
                            save_json=params['save_json'], save_txt=params['save_txt'],
                            single_dragon=params['single_dragon'], only_person=params['only_person'], only_dragon=params['only_dragon'],
                            classify=params['classify'], classify_model=params['classify_model'], verbose=False,
                            node_colors=params['node_colors'], node_size=params['node_size'],
                            line_color=params['line_color'], line_thickness=params['line_thickness']
                        )
                        
                        if gpu_status_placeholder:
                            gpu_status_placeholder.empty()
                        
                        # 更新显示结果
                        preview_placeholder.image(
                            cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                            caption="检测结果",
                            width='stretch'
                        )
                        
                        # 提示用户及时保存
                        st.success("处理完成！请及时保存输出结果，关闭浏览器后文件将自动清除。")
                        
                        # 提供下载选项
                        download_cols = st.columns(4)
                        
                        if params['save_json']:
                            with download_cols[0]:
                                with open(OUTPUT_DIR / "person.json", "rb") as file:
                                    st.download_button(
                                        label="📥 人体JSON",
                                        data=file,
                                        file_name="person_detection.json",
                                        mime="application/json",
                                        width='stretch',
                                        key="download_person_json"
                                    )
                            with download_cols[1]:
                                with open(OUTPUT_DIR / "dragon.json", "rb") as file:
                                    st.download_button(
                                        label="📥 龙骨架JSON",
                                        data=file,
                                        file_name="dragon_detection.json",
                                        mime="application/json",
                                        width='stretch',
                                        key="download_dragon_json"
                                    )
                        
                        if params['save_txt']:
                            with download_cols[2]:
                                with open(OUTPUT_DIR / "person.txt", "rb") as file:
                                    st.download_button(
                                        label="📥 人体TXT",
                                        data=file,
                                        file_name="person_detection.txt",
                                        mime="text/plain",
                                        width='stretch',
                                        key="download_person_txt"
                                    )
                            with download_cols[3]:
                                with open(OUTPUT_DIR / "dragon.txt", "rb") as file:
                                    st.download_button(
                                        label="📥 龙骨架TXT",
                                        data=file,
                                        file_name="dragon_detection.txt",
                                        mime="text/plain",
                                        width='stretch',
                                        key="download_dragon_txt"
                                    )
                        
                        # 图片下载按钮
                        with open(output_img_path, "rb") as file:
                            st.download_button(
                                label="📥 下载检测结果图片",
                                data=file,
                                file_name="detection_result.jpg",
                                mime="image/jpeg",
                                width='stretch',
                                key="download_result_image"
                            )

        # 主内容区域 - 视频模式
        elif current_mode == "video":
            # 初始化视频播放相关状态
            if 'video_played' not in st.session_state:
                st.session_state.video_played = False
                
            uploaded_file = st.file_uploader("选择视频文件", type=['mp4', 'mov', 'avi', 'mkv'], key="upload_video")
            
            if uploaded_file is not None and not st.session_state.analysis_running:
                # 保存上传的视频到临时目录
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    video_path = tmp_file.name
                
                # 显示上传的视频信息
                try:
                    cap_info = cv2.VideoCapture(video_path)
                    if cap_info.isOpened():
                        width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap_info.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                        st.info(f"视频信息: {width}x{height}, {fps:.1f} FPS, {frame_count} 帧")
                    cap_info.release()
                except Exception as e:
                    st.warning(f"无法获取视频信息: {str(e)}")
                
                # 控制按钮
                col1, col2 = st.columns(2)
                with col1:
                    start_button = st.button("开始处理视频", width='stretch', type="primary", key="btn_start_video")
                with col2:
                    stop_button = st.button("终止处理", width='stretch', disabled=True, key="btn_stop_video")
                
                if start_button:
                    st.session_state.analysis_running = True
                    st.session_state.stop_flag = False
                    st.session_state.video_played = False
                    
                    # 获取参数设置
                    params = get_params()
                    
                    # 状态和进度显示
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    def update_status(text, progress=0):
                        status_text.text(text)
                        progress_bar.progress(progress)
                    
                    # 直接在主线程运行视频处理
                    output_video_path = process_video(
                        params['person_model'], params['dragon_model'], video_path, params['confs'], 
                        params['realtime_filter_method'], params['smooth'], params['device'],
                        save_json=params['save_json'], save_txt=params['save_txt'],
                        show_preview=params['show_preview'], single_dragon=params['single_dragon'], 
                        only_person=params['only_person'], only_dragon=params['only_dragon'],
                        classify=params['classify'], classify_model=params['classify_model'],
                        save_video=params['save_video'], verbose=False,
                        node_colors=params['node_colors'], node_size=params['node_size'],
                        line_color=params['line_color'], line_thickness=params['line_thickness'],
                        gpu_monitor=gpu_monitor,
                        status_callback=update_status
                    )
                    
                    st.session_state.analysis_running = False
                    st.session_state.output_video_path = output_video_path
                    st.session_state.video_played = True
                    update_status("视频处理完成！", 1.0)
                    
                    # 显示处理结果
                    if output_video_path and output_video_path.exists():
                        st.subheader("处理结果预览")
                        # 读取视频文件内容
                        try:
                            with open(output_video_path, "rb") as file:
                                video_bytes = file.read()
                                # 检查文件大小
                                if len(video_bytes) < 1024:  # 小于1KB的视频文件可能为空
                                    st.error("生成的视频文件过小，可能为空或损坏")
                                else:
                                    # 使用Streamlit的视频播放器
                                    st.video(video_bytes, format="video/mp4")
                        except Exception as e:
                            st.error(f"无法读取视频文件: {str(e)}")
                        
                        # 验证视频文件
                        cap = cv2.VideoCapture(str(output_video_path))
                        if not cap.isOpened():
                            st.error("生成的视频文件无法打开，可能已损坏")
                        else:
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            st.info(f"视频信息: {frame_count} 帧, {fps:.1f} FPS")
                            cap.release()
                        
                        # 提示用户及时保存
                        st.success("处理完成！请及时保存输出结果，关闭浏览器后文件将自动清除。")
                        
                        # 提供下载选项
                        download_cols = st.columns(4)
                        
                        if params['save_json']:
                            with download_cols[0]:
                                with open(OUTPUT_DIR / "person.json", "rb") as file:
                                    st.download_button(
                                        label="📥 人体JSON",
                                        data=file,
                                        file_name="person_detection.json",
                                        mime="application/json",
                                        width='stretch',
                                        key="download_video_person_json"
                                    )
                            with download_cols[1]:
                                with open(OUTPUT_DIR / "dragon.json", "rb") as file:
                                    st.download_button(
                                        label="📥 龙骨架JSON",
                                        data=file,
                                        file_name="dragon_detection.json",
                                        mime="application/json",
                                        width='stretch',
                                        key="download_video_dragon_json"
                                    )
                        
                        if params['save_txt']:
                            with download_cols[2]:
                                with open(OUTPUT_DIR / "person.txt", "rb") as file:
                                    st.download_button(
                                        label="📥 人体TXT",
                                        data=file,
                                        file_name="person_detection.txt",
                                        mime="text/plain",
                                        width='stretch',
                                        key="download_video_person_txt"
                                    )
                            with download_cols[3]:
                                with open(OUTPUT_DIR / "dragon.txt", "rb") as file:
                                    st.download_button(
                                        label="📥 龙骨架TXT",
                                        data=file,
                                        file_name="dragon_detection.txt",
                                        mime="text/plain",
                                        width='stretch',
                                        key="download_video_dragon_txt"
                                    )
                        
                        # 视频下载按钮
                        with open(output_video_path, "rb") as file:
                            st.download_button(
                                label="📥 下载处理后的视频",
                                data=file,
                                file_name="processed_video.mp4",
                                mime="video/mp4",
                                width='stretch',
                                key="download_processed_video"
                            )
                    elif output_video_path:
                        st.error(f"视频处理失败，文件未生成: {output_video_path}")
                    else:
                        st.error("视频处理失败，未生成输出文件")

        # 主内容区域 - 摄像头模式
        elif current_mode == "camera":
            # 获取检测参数
            params = get_params()

            # 调用新的 WebRTC 摄像头处理逻辑
            process_camera_stream(params, gpu_monitor)

    # 参数设置独立页面
    elif st.session_state.current_tab == "参数设置":
        st.markdown("<h1 style='text-align: center;'>参数设置</h1>", unsafe_allow_html=True)
        st.write("在这里配置系统的各项参数，设置将应用于所有检测模式。")
        
        # 使用session_state存储参数，确保在页面间保持一致
        if 'params_initialized' not in st.session_state:
            # 初始化参数
            st.session_state.use_gpu = gpu_monitor.use_gpu
            st.session_state.single_dragon = True
            st.session_state.only_person = False
            st.session_state.only_dragon = False
            st.session_state.classify = True
            st.session_state.show_preview = True
            st.session_state.save_video = True
            st.session_state.save_json = False
            st.session_state.save_txt = False
            st.session_state.person_model = "yolov8m-pose.pt"
            st.session_state.dragon_model = "YoGon-Pose-v2.pt"
            st.session_state.classify_model = "YoGon-Clas-v2.pth"
            st.session_state.node_size = 10
            st.session_state.line_thickness = 6
            st.session_state.person_conf = 0.3
            st.session_state.dragon_conf = 0.3
            st.session_state.person_kpt_conf = 0.5
            st.session_state.dragon_kpt_conf = 0.5
            st.session_state.realtime_filter_method = "none"
            st.session_state.smooth = "none"
            
            # 初始化节点颜色
            hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            for i in range(9):
                default_color = '#%02x%02x%02x' % (
                    DEFAULT_NODE_COLORS[i][2], 
                    DEFAULT_NODE_COLORS[i][1], 
                    DEFAULT_NODE_COLORS[i][0]
                )
                st.session_state[f"node_color_{i}"] = default_color
            
            st.session_state.line_color = "#C8C8C8"
            st.session_state.params_initialized = True
        
        # 设备配置
        with st.expander("💻 设备配置", expanded=True):
            st.session_state.use_gpu = st.checkbox(
                "使用GPU加速", 
                value=st.session_state.use_gpu, 
                disabled=not gpu_monitor.use_gpu,
                key="cb_use_gpu"
            )
            device = torch.device('cuda' if (st.session_state.use_gpu and gpu_monitor.use_gpu) else 'cpu')
            
            # 显示设备信息
            if device.type == 'cuda':
                st.success(f"使用GPU: {gpu_monitor.gpu_name}")
                st.info(f"总显存: {gpu_monitor.memory_total:.2f} GB")
            else:
                st.warning("使用CPU进行计算，可能较慢")
        
        # 检测选项
        with st.expander("🔍 检测选项", expanded=True):
            def on_detection_param_change():
                if st.session_state.analysis_running:
                    st.warning("检测参数已修改，当前分析将终止并需要重新开始")
                    st.session_state.analysis_running = False
                    st.session_state.stop_flag = True
            
            st.session_state.single_dragon = st.checkbox(
                "仅检测单条龙", 
                value=st.session_state.single_dragon, 
                on_change=on_detection_param_change,
                key="cb_single_dragon"
            )
            st.session_state.only_person = st.checkbox(
                "仅检测人体", 
                value=st.session_state.only_person, 
                on_change=on_detection_param_change,
                key="cb_only_person"
            )
            st.session_state.only_dragon = st.checkbox(
                "仅检测龙骨架", 
                value=st.session_state.only_dragon, 
                on_change=on_detection_param_change,
                key="cb_only_dragon"
            )
            st.session_state.classify = st.checkbox(
                "启用动作分类", 
                value=st.session_state.classify, 
                on_change=on_detection_param_change,
                key="cb_classify"
            )

        # 输出选项
        with st.expander("💾 输出选项", expanded=True):
            def on_output_param_change():
                if st.session_state.analysis_running:
                    st.warning("输出参数已修改，当前分析将终止并需要重新开始")
                    st.session_state.analysis_running = False
                    st.session_state.stop_flag = True
            
            st.session_state.show_preview = st.checkbox(
                "实时预览", 
                value=st.session_state.show_preview, 
                on_change=on_output_param_change,
                key="cb_show_preview"
            )
            st.session_state.save_video = st.checkbox(
                "保存视频", 
                value=st.session_state.save_video, 
                on_change=on_output_param_change,
                key="cb_save_video"
            )
            st.session_state.save_json = st.checkbox(
                "保存JSON文件", 
                value=st.session_state.save_json, 
                on_change=on_output_param_change,
                key="cb_save_json"
            )
            st.session_state.save_txt = st.checkbox(
                "保存TXT文件", 
                value=st.session_state.save_txt, 
                on_change=on_output_param_change,
                key="cb_save_txt"
            )
        
        # 模型配置
        with st.expander("🤖 模型配置", expanded=True):
            def on_model_param_change():
                if st.session_state.analysis_running:
                    st.warning("模型参数已修改，当前分析将终止并需要重新开始")
                    st.session_state.analysis_running = False
                    st.session_state.stop_flag = True
            
            st.session_state.person_model = st.text_input(
                "人体姿态模型路径", 
                value=st.session_state.person_model, 
                on_change=on_model_param_change,
                key="txt_person_model"
            )
            st.session_state.dragon_model = st.text_input(
                "龙骨架模型路径", 
                value=st.session_state.dragon_model, 
                on_change=on_model_param_change,
                key="txt_dragon_model"
            )
            st.session_state.classify_model = st.text_input(
                "动作分类模型路径", 
                value=st.session_state.classify_model, 
                on_change=on_model_param_change,
                key="txt_classify_model"
            )


        # 样式自定义配置
        with st.expander("🎨 样式自定义", expanded=False):
            def on_style_param_change():
                if st.session_state.analysis_running:
                    st.warning("样式参数已修改，当前分析将终止并需要重新开始")
                    st.session_state.analysis_running = False
                    st.session_state.stop_flag = True
            
            # 节点样式
            st.subheader("节点样式")
            st.session_state.node_size = st.slider(
                "节点大小", 
                1, 20, st.session_state.node_size, 
                on_change=on_style_param_change,
                key="slider_node_size"
            )
            
            # 每个节点单独的颜色选择
            hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            for i in range(9):  # 1-9节点
                st.session_state[f"node_color_{i}"] = st.color_picker(
                    f"节点 {i+1} 颜色", 
                    st.session_state[f"node_color_{i}"], 
                    key=f"cp_node_{i}",
                    on_change=on_style_param_change
                )
            
            # 连线样式
            st.subheader("连线样式")
            st.session_state.line_thickness = st.slider(
                "连线粗细", 
                1, 10, st.session_state.line_thickness, 
                on_change=on_style_param_change,
                key="slider_line_thickness"
            )
            st.session_state.line_color = st.color_picker(
                "连线颜色", 
                st.session_state.line_color, 
                on_change=on_style_param_change,
                key="cp_line_color"
            )

        # 置信度配置
        with st.expander("🎯 置信度阈值", expanded=False):
            def on_conf_param_change():
                if st.session_state.analysis_running:
                    st.warning("置信度参数已修改，当前分析将终止并需要重新开始")
                    st.session_state.analysis_running = False
                    st.session_state.stop_flag = True
            
            st.session_state.person_conf = st.slider(
                "人体检测置信度", 
                0.1, 1.0, st.session_state.person_conf, 0.05, 
                on_change=on_conf_param_change,
                key="slider_person_conf"
            )
            st.session_state.dragon_conf = st.slider(
                "龙骨架检测置信度", 
                0.1, 1.0, st.session_state.dragon_conf, 0.05, 
                on_change=on_conf_param_change,
                key="slider_dragon_conf"
            )
            st.session_state.person_kpt_conf = st.slider(
                "人体关键点置信度", 
                0.1, 1.0, st.session_state.person_kpt_conf, 0.05, 
                on_change=on_conf_param_change,
                key="slider_person_kpt_conf"
            )
            st.session_state.dragon_kpt_conf = st.slider(
                "龙骨架关键点置信度", 
                0.1, 1.0, st.session_state.dragon_kpt_conf, 0.05, 
                on_change=on_conf_param_change,
                key="slider_dragon_kpt_conf"
            )

        # 滤波配置
        with st.expander("📊 滤波设置", expanded=False):
            def on_filter_param_change():
                if st.session_state.analysis_running:
                    st.warning("滤波参数已修改，当前分析将终止并需要重新开始")
                    st.session_state.analysis_running = False
                    st.session_state.stop_flag = True
            
            st.session_state.realtime_filter_method = st.selectbox(
                "实时滤波方法",
                ["none", "ema", "kalman"],
                index=["none", "ema", "kalman"].index(st.session_state.realtime_filter_method),
                on_change=on_filter_param_change,
                key="select_realtime_filter"
            )
            st.session_state.smooth = st.selectbox(
                "视频平滑方案",
                ["none", "mean", "weighted_mean", "ewm"],
                index=["none", "mean", "weighted_mean", "ewm"].index(st.session_state.smooth),
                on_change=on_filter_param_change,
                key="select_smooth"
            )
        
        if st.button("保存参数设置", width='stretch', type="primary", key="btn_save_params"):
            st.success("参数设置已保存！")

    # 开发团队页面
    elif st.session_state.current_tab == "开发团队":
        st.markdown("<h1 style='text-align: center;'>开发团队</h1>", unsafe_allow_html=True)
        
        # 团队介绍
        st.subheader("项目简介")
        st.write("""
        点睛AI（DragonJot）是国内首个专注于舞龙运动的AI智能评判与训练辅助系统。针对传统舞龙赛事评判主观性强、训练反馈缺乏量化依据、教学资源不足等行业痛点，我们创新提出"人龙一体"协同动作识别模型。该系统基于YOLO姿态识别模型与CNN动作分类算法，结合自主构建的舞龙骨骼数据集，实现了对人体姿态与龙形轨迹的同步识别与分析。

        系统支持多路视频输入（包括手机直播与本地视频），具备实时动作捕捉、动作分类、实时分析与量化评估等核心功能，全面覆盖赛事评判、专业训练与大众教学三大应用场景。

        Open DragonJot作为点睛AI的开源产品，提供核心功能的体验版，包含人体姿态识别、龙运动轨迹识别和舞龙动作分类功能。开源代码已发布于GitHub，产品预览网页部署在Streamlit平台，欢迎体验与交流。
        """)
        
        # 团队成员
        st.subheader("核心开发团队")
        
        # 成员1
        col1, col2 = st.columns([1, 9])  # 调整比例使头像区域更小
        with col1:
            try:
                st.image(str(STATIC_DIR / "avatars" / "HC.jpg"), caption="项目负责人", width='stretch')
            except:
                st.image("https://via.placeholder.com/100", caption="项目负责人", width='stretch')
        with col2:
            st.markdown("""
            ### 霍畅 - 项目负责人
            负责团队统筹与核心规划，提出项目核心创意，主导应用开发与落地。
            """)
        
        # 成员2
        col1, col2 = st.columns([1, 9])
        with col1:
            try:
                st.image(str(STATIC_DIR / "avatars" / "LZC.jpg"), caption="算法工程师", width='stretch')
            except:
                st.image("https://via.placeholder.com/100", caption="算法工程师", width='stretch')
        with col2:
            st.markdown("""
            ### 卢子诚 - 算法工程师
            负责技术调研、AI模型训练与优化，以及核心算法的开发落地。
            """)
        
        # 成员3
        col1, col2 = st.columns([1, 9])
        with col1:
            try:
                st.image(str(STATIC_DIR / "avatars" / "FZY.jpg"), caption="UI/UX设计师", width='stretch')
            except:
                st.image("https://via.placeholder.com/100", caption="UI/UX设计师", width='stretch')
        with col2:
            st.markdown("""
            ### 方子嫣 - UI/UX设计师
            负责产品创意策划、用户界面与体验设计，塑造产品的整体视觉风格。
            """)
        
        # 成员4
        col1, col2 = st.columns([1, 9])
        with col1:
            try:
                st.image(str(STATIC_DIR / "avatars" / "DWY.jpg"), caption="文案策划", width='stretch')
            except:
                st.image("https://via.placeholder.com/100", caption="文案策划", width='stretch')
        with col2:
            st.markdown("""
            ### 丁闻玥 - 创意内容总监
            负责市场调研，并总负责产品报告、落地推广方案等关键文稿的撰写。
            """)

        # 成员5
        col1, col2 = st.columns([1, 9])
        with col1:
            try:
                st.image(str(STATIC_DIR / "avatars" / "ZMH.jpg"), caption="数据工程师", width='stretch')
            except:
                st.image("https://via.placeholder.com/100", caption="数据工程师", width='stretch')
        with col2:
            st.markdown("""
            ### 张明涵 - 数据工程师
            主导舞龙数据集的规划、采集与构建工作，为模型训练提供核心数据基础。
            """)

        # 成员6
        col1, col2 = st.columns([1, 9])
        with col1:
            try:
                st.image(str(STATIC_DIR / "avatars" / "ZKX.jpg"), caption="产品宣传", width='stretch')
            except:
                st.image("https://via.placeholder.com/100", caption="产品宣传", uwidth='stretch')
        with col2:
            st.markdown("""
            ### 赵康西 – 产品可视化专员
            负责产品演示视频的剪辑制作、数据收集和部分前端页面开发。
            """)
        
        # 联系方式
        st.subheader("联系方式")
        st.write("""
        如果您对"点睛AI"项目感兴趣，或有任何疑问、建议与合作意向，欢迎通过以下方式与我们取得联系：
        
        邮箱：22307110080@m.fudan.edu.cn
        
        项目地址：xxxx
        
        期待您的反馈与交流！
        """)
        st.write("   ")
        st.write("   ")
        st.write("   ")
        st.write("   ")
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        

        # 致谢（居中显示在末尾）
        st.markdown("""
    <div style='text-align: center; margin: 20px 0;'>
        <h3>致 谢</h3>
        <p style='margin: 15px 0; line-height: 1.8;'>
            一路走来，<br>"点睛AI"的每一步成长都离不开众多支持者的鼎力相助。<br>在此，我们谨向所有为本项目<br>
            倾注心血、给予指导的机构与个人，<br>致以最诚挚的敬意与最衷心的感谢！
        </p>
        <p style='margin: 15px 0; line-height: 1.8;'>
            特别鸣谢 <strong style='font-size: 1.1em;'>复旦大学体育发展部的徐燕勤老师</strong>，以及所有<br>
            <strong style='font-size: 1.1em;'>第一届"复旦AI+体育创变营"</strong> 的老师与同学们——<br>
            是你们搭建的创新平台，让这个源于对传统文化热爱的项目种子得以萌发、成长。
        </p>
        <p style='margin: 15px 0; line-height: 1.8;'>
            特别感谢 <strong style='font-size: 1.1em;'>上海龙狮协会朱俊鹏教练</strong> 的专业引领——<br>
            您不仅精准点出行业痛点，更无私分享宝贵的行业资源，为我们的技术落地指明了方向。
        </p>
        <p style='margin: 15px 0; line-height: 1.8;'>
            深深感激 <strong style='font-size: 1.1em;'>复旦龙狮协会</strong> 的同学们——<br>
            你们用专业的表演与热情的投入，为项目提供了高质量的数据集与演示素材，<br>
            成为"点睛AI"能够栩栩如生的核心基石。
        </p>
        <p style='margin: 15px 0; line-height: 1.8;'>
            因为有你们，传统文化与现代科技的碰撞才如此精彩。<br>
            这份支持，将激励我们继续用技术赋能传统，让舞龙文化焕发新的生机！
        </p>
    </div>
    """, unsafe_allow_html=True)

# 参数获取函数，供各模块使用统一的参数设置
def get_params():
    # 确保参数已初始化
    if 'params_initialized' not in st.session_state:
        # 避免递归调用main()，直接初始化参数
        st.session_state.use_gpu = gpu_monitor.use_gpu
        st.session_state.single_dragon = True
        st.session_state.only_person = False
        st.session_state.only_dragon = False
        st.session_state.classify = True
        st.session_state.show_preview = True
        st.session_state.save_video = True
        st.session_state.save_json = False
        st.session_state.save_txt = False
        st.session_state.person_model = "yolov8m-pose.pt"
        st.session_state.dragon_model = "YoGon-Pose-v2.pt"
        st.session_state.classify_model = "YoGon-Clas-v2.pth"
        st.session_state.node_size = 10
        st.session_state.line_thickness = 6
        st.session_state.person_conf = 0.3
        st.session_state.dragon_conf = 0.3
        st.session_state.person_kpt_conf = 0.5
        st.session_state.dragon_kpt_conf = 0.5
        st.session_state.realtime_filter_method = "none"
        st.session_state.smooth = "none"
        
        # 初始化节点颜色
        hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        for i in range(9):
            default_color = '#%02x%02x%02x' % (
                DEFAULT_NODE_COLORS[i][2], 
                DEFAULT_NODE_COLORS[i][1], 
                DEFAULT_NODE_COLORS[i][0]
            )
            st.session_state[f"node_color_{i}"] = default_color
        
        st.session_state.line_color = "#C8C8C8"
        st.session_state.params_initialized = True
    
    # 设备配置
    device = torch.device(
        'cuda' if (st.session_state.use_gpu and gpu_monitor.use_gpu) else 'cpu'
    )
    
    # 置信度参数
    confs = [
        st.session_state.person_conf, 
        st.session_state.dragon_conf, 
        st.session_state.person_kpt_conf, 
        st.session_state.dragon_kpt_conf
    ]
    
    # 节点颜色处理
    hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    node_colors = []
    for i in range(9):
        r, g, b = hex_to_rgb(st.session_state[f"node_color_{i}"])
        node_colors.append((b, g, r))  # 转换为BGR格式
    
    # 连线颜色处理
    line_r, line_g, line_b = hex_to_rgb(st.session_state.line_color)
    line_color = (line_b, line_g, line_r)  # 转换为BGR格式
    
    return {
        'device': device,
        'single_dragon': st.session_state.single_dragon,
        'only_person': st.session_state.only_person,
        'only_dragon': st.session_state.only_dragon,
        'classify': st.session_state.classify,
        'show_preview': st.session_state.show_preview,
        'save_video': st.session_state.save_video,
        'save_json': st.session_state.save_json,
        'save_txt': st.session_state.save_txt,
        'person_model': st.session_state.person_model,
        'dragon_model': st.session_state.dragon_model,
        'classify_model': st.session_state.classify_model,
        'node_size': st.session_state.node_size,
        'line_thickness': st.session_state.line_thickness,
        'confs': confs,
        'realtime_filter_method': st.session_state.realtime_filter_method,
        'smooth': st.session_state.smooth,
        'node_colors': node_colors,
        'line_color': line_color
    }

if __name__ == "__main__":
    main()
