import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Live Object Detection",
    page_icon="🎥",
    layout="wide"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #a78bfa;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(
    '<div class="title">🎥 Live Object Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-time AI detection using YOLOv8</div>',
    unsafe_allow_html=True
)

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Settings")

confidence = st.sidebar.slider(
    "Confidence",
    0.0,
    1.0,
    0.25
)

model_option = st.sidebar.selectbox(
    "Model",
    ["yolov8n.pt", "yolov8s.pt"]
)

alert_object = st.sidebar.selectbox(
    "🚨 Alert Object",
    ["person", "car", "motorcycle", "dog", "chair", "bottle"]
)

save_frames = st.sidebar.checkbox("💾 Save Frames")

# -------------------- SAVE DIRECTORY --------------------
SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_option)

# -------------------- INFO --------------------
st.info("📷 Allow camera access to start detection")

# -------------------- CALLBACK --------------------
def video_frame_callback(frame):

    # Convert frame
    img = frame.to_ndarray(format="bgr24")

    # YOLO prediction
    results = model.predict(
        img,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Draw YOLO boxes
    annotated_frame = result.plot()

    boxes = result.boxes

    object_count = 0
    alert_count = 0

    # -------------------- DETECTION --------------------
    if boxes is not None:

        object_count = len(boxes)

        for box in boxes:

            cls_id = int(box.cls[0])

            detected_name = model.names[cls_id].lower()

            # Check alert object
            if detected_name == alert_object.lower():

                alert_count += 1

    # -------------------- ALERT DISPLAY --------------------
    if alert_count > 0:

        cv2.putText(
            annotated_frame,
            f"ALERT: {alert_object.upper()} DETECTED!",
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"COUNT: {alert_count}",
            (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            annotated_frame,
            "NO ALERT",
            (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -------------------- OBJECT COUNT --------------------
    cv2.putText(
        annotated_frame,
        f"Objects: {object_count}",
        (0, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

    # -------------------- SAVE FRAME --------------------
    if save_frames and object_count > 0:

        # Unique filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        milliseconds = int((time.time() % 1) * 1000)

        filename = os.path.join(
            SAVE_DIR,
            f"frame_{timestamp}-{milliseconds}.jpg"
        )

        # Save image
        success = cv2.imwrite(filename, annotated_frame)

        # Debug message
        if success:
            print(f"Saved: {filename}")
        else:
            print("Failed to save frame")

    # Return frame
    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )

# -------------------- WEBRTC STREAM --------------------
webrtc_streamer(
    key="object-detection",

    video_frame_callback=video_frame_callback,

    async_processing=True,

    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },

    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

# -------------------- FOOTER --------------------
st.success("🟢 Camera Ready")