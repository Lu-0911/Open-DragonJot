import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import cv2

# ---------- 你的常量/工具函数（从 app.py 拷过来） ----------
from app import (                     # 直接复用 app.py 里已实现的东西
    DRAGON_KEYPOINT_NAMES,
    DRAGON_SKELETON,
    DEFAULT_NODE_COLORS,
    DEFAULT_LINE_COLOR,
    gpu_monitor,
    get_params,
    put_chinese_text,
    classify_action,
    build_class_inputs,
    EMAFilter,
    KalmanFilterWrapper
)

# ---------- WebRTC 回调 ----------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")          # 浏览器采集到的帧
    params = get_params()                           # 读取你在侧边栏里保存的参数
    device = params['device']

    # 1. 加载模型（缓存避免每次重载）
    if "model_person" not in st.session_state:
        st.session_state.model_person = YOLO(params['person_model']).to(device)
    if "model_dragon" not in st.session_state:
        st.session_state.model_dragon = YOLO(params['dragon_model']).to(device)
    if "classify_model_obj" not in st.session_state and params['classify']:
        ckpt = torch.load(Path("model")/params['classify_model'], map_location=device)
        from app import PoseCNN
        model = PoseCNN(num_classes=len(ckpt['classes'])).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        st.session_state.classify_model_obj = model
        st.session_state.classes = ckpt['classes']

    # 2. 推理
    person_results = st.session_state.model_person(img, conf=params['confs'][0], verbose=False)
    dragon_results = st.session_state.model_dragon(img, conf=params['confs'][1], verbose=False)

    # 3. 画关键点 + 骨架（直接复用你 app.py 里的代码）
    out = img.copy()
    if person_results:
        out = person_results[0].plot(boxes=False)
    if dragon_results and dragon_results[0].keypoints is not None:
        kpts = dragon_results[0].keypoints.xy.cpu().numpy()
        conf = dragon_results[0].keypoints.conf.cpu().numpy()
        if params['single_dragon'] and len(kpts) > 0:
            best = int(np.argmax(dragon_results[0].boxes.conf.cpu().numpy()))
            kpts, conf = kpts[best:best+1], conf[best:best+1]
        for i, kp_set in enumerate(kpts):
            for j, ((x, y), c) in enumerate(zip(kp_set, conf[i])):
                if c > params['confs'][3]:      # dragon_kpt_conf
                    color = params['node_colors'][j % len(params['node_colors'])]
                    cv2.circle(out, (int(x), int(y)), params['node_size'], color, -1)
            for (a, b) in DRAGON_SKELETON:
                if conf[i][a] > params['confs'][3] and conf[i][b] > params['confs'][3]:
                    pt1 = tuple(map(int, kp_set[a]))
                    pt2 = tuple(map(int, kp_set[b]))
                    cv2.line(out, pt1, pt2, params['line_color'], params['line_thickness'])

    # 4. 动作分类
    if params['classify'] and dragon_results:
        try:
            p_arr, d_arr = build_class_inputs(person_results, dragon_results, (out.shape[1], out.shape[0]))
            label = classify_action(st.session_state.classify_model_obj,
                                    st.session_state.classes,
                                    p_arr, d_arr, device)
            out = put_chinese_text(out, f"Action: {label}")
        except:
            pass

    return av.VideoFrame.from_ndarray(out, format="bgr24")

# ---------- Streamlit 页面 ----------
st.markdown("<h2 style='text-align:center'>📹 浏览器摄像头实时检测</h2>", unsafe_allow_html=True)
webrtc_streamer(
    key="dragon-camera",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
