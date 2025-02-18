# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import json
import subprocess
import torch
import cv2
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:
    def __init__(self, **kwargs):
        check_requirements("streamlit>=1.29.0")

        # Load daftar CCTV dari JSON
        with open("./cctv_data.json") as f:
            self.list_cctv = json.load(f)

        # Inisialisasi Streamlit
        self.source = None
        self.city = None
        self.cctv_name = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None
        self.model_path = kwargs.get("model", None)

    def check_gpu(self):
        """Mengecek apakah GPU tersedia dan menampilkan nvidia-smi."""
        if torch.cuda.is_available():
            gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
            st.text("GPU is available ✅")
            st.text(gpu_info)
        else:
            st.warning("No GPU detected. Ensure you run the container with --gpus all.")

    def web_ui(self):
        """Konfigurasi tampilan UI Streamlit."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
        st.set_page_config(page_title="Computer Vision Playground", layout="wide")
        st.markdown(menu_style_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Sidebar konfigurasi pengguna."""
        st.sidebar.title("User Configuration")
        self.source = st.sidebar.selectbox("Video", ("webcam", "video", "cctv"))

        if self.source == "cctv":
            self.city = st.sidebar.selectbox("City", ("BANDUNG", "YOGYAKARTA"))
            cctv_names = [loc["cctv_name"] for loc in self.list_cctv if loc["city"] == self.city]
            self.cctv_name = st.sidebar.selectbox("CCTV Location", cctv_names)

        self.enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No"))
        self.conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
        self.iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

        col1, col2 = st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        """Mengatur sumber video yang akan diproses."""
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("playground.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "playground.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0
        elif self.source == "cctv":
            self.vid_file_name = [loc["cctv_link"] for loc in self.list_cctv if (loc["city"] == self.city and loc["cctv_name"] == self.cctv_name)][0]

    def configure(self):
        """Konfigurasi model dan memuat model YOLO dengan GPU."""
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = st.sidebar.selectbox("Model", available_models)

        with st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt").to("cuda")
            class_names = list(self.model.names.values())

        st.success("Model loaded successfully!")

        if self.source == "cctv":
            st.success(self.cctv_name)

        selected_classes = st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        """Menjalankan proses inference dengan Streamlit."""
        self.web_ui()
        self.sidebar()
        self.check_gpu()
        self.source_upload()
        self.configure()

        if st.sidebar.button("Start"):
            stop_button = st.button("Stop")
            cap = cv2.VideoCapture(self.vid_file_name)

            if not cap.isOpened():
                st.error("Could not open video source.")
                return

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.warning("Failed to read frame. Please check the video source.")
                    break

                if self.enable_trk == "Yes":
                    results = self.model.track(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True)
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_frame = results[0].plot()

                if stop_button:
                    cap.release()
                    st.stop()

                self.org_frame.image(frame, channels="BGR")
                self.ann_frame.image(annotated_frame, channels="BGR")

            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None

    Inference(model=model).inference()
