import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import time
import math
import json
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# ── Module-level imports so recv() is not paying import cost every frame ──────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.digit_classifier import DigitClassifier
from ai.detector import process_frame, debug_frame, process_multi_frame
from ai.perspective import find_board
from ai.utils.utils import display_numbers_on_board

# Page configuration
st.set_page_config(
    page_title="Industrial AI Sudoku Solver",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .animated-header {
        background: linear-gradient(270deg, #1f4037, #99f2c8, #1f4037);
        background-size: 600% 600%;
        animation: AnimationName 10s ease infinite;
        padding: 20px; border-radius: 10px;
        text-align: center; color: #fff; margin-bottom: 30px;
    }
    @keyframes AnimationName {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }
    .metric-card {
        background-color: #262730; padding: 20px; border-radius: 10px;
        border-left: 5px solid #00f2fe;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00f2fe; }
    .metric-title { font-size: 1rem; color: #A0AEC0; }
</style>
""", unsafe_allow_html=True)

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Model loaded ONCE, cached across all reruns ────────────────────────────────
# FIX #1: @st.cache_resource guarantees exactly one load_model() call per server
# lifetime. The webcam processor reuses this exact object — no second TF load.
@st.cache_resource
def load_classifier():
    return DigitClassifier(model_path='models/sudoku_digit_model.h5')

global_classifier = load_classifier()

# Sidebar — FIX #2: use local emoji instead of remote HTTP image fetch
st.sidebar.markdown("## 🤖")
st.sidebar.title("AI Navigation")
page = st.sidebar.radio("Go to",
    ["Dashboard", "Upload Sudoku", "Live Webcam Solver", "Results & Analytics", "Settings"]
)

def add_to_history(success, t):
    st.session_state.history.append({"timestamp": time.time(), "success": success, "processing_time": t})

# ─── Dashboard ───────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown('<div class="animated-header"><h1>🤖 Enterprise AI Sudoku Platform</h1></div>', unsafe_allow_html=True)

    total = len(st.session_state.history)
    ok = sum(1 for h in st.session_state.history if h['success'])
    avg = np.mean([h['processing_time'] for h in st.session_state.history]) if total else 0

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-title">Total Processed</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-title">Success Rate</div><div class="metric-value">{(ok/total*100) if total else 0:.1f}%</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-title">Avg Time</div><div class="metric-value">{avg:.3f}s</div></div>', unsafe_allow_html=True)

    st.markdown("### System Status")
    st.info("🟢 Computer Vision Pipeline: Online")
    if global_classifier.model:
        st.success("🟢 Deep Learning Model: Loaded")
    else:
        st.error("🔴 Deep Learning Model: Offline")
    st.success("🟢 Backtracking Solver: Online")

# ─── Upload Sudoku ───────────────────────────────────────────────────────────
elif page == "Upload Sudoku":
    st.title("📤 Image Upload Processor")
    uploaded = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300, caption="Source Image")

        if st.button("Run Visual Diagnostics", type="primary"):
            st.error("VISUAL DIAGNOSTICS MODE ACTIVE")
            print("VISUAL DIAGNOSTICS STARTED")
            import os
            # Ensure debug directory exists
            os.makedirs("debug", exist_ok=True)
            
            t_start = time.perf_counter()
            with st.spinner("Running strict visual debugging pipeline..."):
                result = debug_frame(img, global_classifier)
                elapsed = time.perf_counter() - t_start

            stage = result.get("stage", "UNKNOWN")
            st.markdown(f"### Pipeline Stage Reached: `{stage}`  —  ⏱ `{elapsed:.2f}s`")

            if stage != "DONE" and stage != "SOLVED" and stage != "OCR_VALIDATION":
                st.error(f"❌ FAILED AT STAGE: **{stage}** — {result.get('error', 'Unknown')}")
                st.stop()

            st.markdown("---")
            st.markdown("## 🔍 VISUAL DEBUGGING PIPELINE")

            # 1. Original
            if os.path.exists("debug/01_original.jpg"):
                st.markdown("### 1. Original Image")
                st.image("debug/01_original.jpg", use_container_width=True)

            # 2. Detected Contour
            if os.path.exists("debug/02_detected_contour.jpg"):
                st.markdown("### 2. Detected Sudoku Contour (RED)")
                st.image("debug/02_detected_contour.jpg", use_container_width=True)

            # 3. Corner Points
            if os.path.exists("debug/03_corner_points.jpg"):
                st.markdown("### 3. Perspective Corner Points")
                st.image("debug/03_corner_points.jpg", use_container_width=True)

            # 4. Warped Board
            if os.path.exists("debug/04_warped_board.jpg"):
                st.markdown("### 4. Perspective Warped Board")
                st.image("debug/04_warped_board.jpg", use_container_width=True)

            # 5. Grid Overlay
            if os.path.exists("debug/05_grid_overlay.jpg"):
                st.markdown("### 5. Calculated 9x9 Grid Lines")
                st.image("debug/05_grid_overlay.jpg", use_container_width=True)

            # 6. Cell Numbering
            if os.path.exists("debug/06_cell_numbering.jpg"):
                st.markdown("### 6. Flattened Cell Indexing (0-80)")
                st.image("debug/06_cell_numbering.jpg", use_container_width=True)

            # 7. Digit Contours
            if os.path.exists("debug/07_digit_contours.jpg"):
                st.markdown("### 7. Detected Digits (Bounding Box & Area %)")
                st.image("debug/07_digit_contours.jpg", use_container_width=True)

            # 8. OCR Matrix Dump
            if os.path.exists("debug/08_ocr_matrix.txt"):
                st.markdown("### 8. Final OCR Matrix")
                with open("debug/08_ocr_matrix.txt", "r") as f:
                    st.code(f.read(), language="json")



            st.markdown("---")
            st.markdown("### 🔬 Cell-by-Cell Extraction Details")
            cell_details = result.get("cell_details", [])
            orig_conflicts = result.get("original_conflicts", [])
            if cell_details:
                cols = st.columns(6)
                displayed = 0
                for d in cell_details:
                    if not d["is_empty"]:
                        with cols[displayed % 6]:
                            st.image(f"debug/cells/cell_{d['index']:02d}.png", width=60)
                            
                            is_conflict = d['index'] in orig_conflicts
                            color = "red" if is_conflict else "green"
                            st.markdown(f"**Cell {d['index']}** <span style='color:{color}'>{'[CONFLICT]' if is_conflict else ''}</span>", unsafe_allow_html=True)
                            
                            st.write(f"Digit: **{d['predicted']}**")
                            st.write(f"Conf: `{d['confidence']*100:.1f}%`")
                            
                            top3 = d.get('top3', [])
                            if top3:
                                st.caption("Top 3:")
                                for t_dig, t_conf in top3:
                                    st.caption(f"- {t_dig}: {t_conf*100:.1f}%")
                                    
                        displayed += 1
                        
            # FIX: 4 — Update frontend Step 9 display
            st.markdown("---")
            st.markdown("## ✅ Step 9. Solution")
            
            solved        = result.get("solved_board")
            board_used    = result.get("board", [])
            stage         = result.get("stage", "")
            was_corrected = result.get("ocr_conflicts_resolved", False)
            orig_errors   = result.get("original_errors", [])
            orig_conflicts = result.get("original_conflicts", [])
          
            if stage == "SOLVED" and solved and len(solved) == 81:
          
                # If OCR was auto-corrected, show a soft warning (not an error)
                if was_corrected and orig_errors:
                    with st.expander(
                        f"⚠️ {len(orig_errors)} OCR conflict(s) were auto-corrected "
                        f"before solving — click to see details", expanded=False):
                        for err in orig_errors:
                            st.warning(err)
                        st.info(
                            f"Cells zeroed (low-confidence): {orig_conflicts}. "
                            f"The solver filled these from scratch."
                        )
          
                # ── PRIMARY OUTPUT: Clean rendered grid ───────────────────────
                solution_img = result.get("solution_grid_img")
                if solution_img is not None:
                    st.markdown("### 🧩 Solved Sudoku")
                    st.image(
                        cv2.cvtColor(solution_img, cv2.COLOR_BGR2RGB),
                        caption="⬛ Black = original clues   🔵 Blue = solver-filled",
                        use_container_width=False,
                        width=590
                    )
                
                # ── SECONDARY: Side-by-side text grids ────────────────────────
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Detected Board**")
                    lines = ""
                    for r in range(9):
                        row = board_used[r*9:(r+1)*9]
                        sep = "  " 
                        lines += sep.join(str(x) if x != 0 else "·" for x in row)
                        if r in (2, 5):
                            lines += "\n" + "─" * 21
                        lines += "\n"
                    st.code(lines, language="text")
          
                with col2:
                    st.markdown("**🏆 Solved Board**")
                    lines = ""
                    for r in range(9):
                        orig_row   = board_used[r*9:(r+1)*9]
                        solved_row = solved[r*9:(r+1)*9]
                        parts = []
                        for i, x in enumerate(solved_row):
                            parts.append(f"[{x}]" if orig_row[i] == 0 else f" {x} ")
                        lines += " ".join(parts)
                        if r in (2, 5):
                            lines += "\n" + "─" * 27
                        lines += "\n"
                    st.code(lines, language="text")
          
                # ── Stats bar ─────────────────────────────────────────────────
                clues  = sum(1 for x in board_used if x != 0)
                filled = 81 - clues
                st.success(
                    f"✅ Puzzle solved!   "
                    f"Clues given: {clues}   |   "
                    f"Cells filled by solver: {filled}   |   "
                    f"Total: 81/81"
                )
                add_to_history(True, elapsed)
          
            else:
                # Truly unresolvable
                val_errors = result.get("validation_errors", [])
                conflicts  = result.get("conflicts", set())
                st.error("🚨 OCR produced conflicts that could not be auto-resolved.")
                for err in val_errors:
                    st.warning(err)
                if conflicts:
                    st.markdown(f"**Conflicting cell indices:** `{sorted(conflicts)}`")
                st.error(
                    "🛑 Solver blocked. Suggestions:\n"
                    "- Use a flatter, more evenly-lit image\n"
                    "- Ensure the full grid border is visible\n"
                    "- Avoid glare or shadows over digits"
                )
                add_to_history(False, elapsed)

# ─── Webcam ──────────────────────────────────────────────────────────────────
elif page == "Live Webcam Solver":
    st.title("🎥 Real-Time Inference Engine")
    st.markdown("Hold your Sudoku grid steady within the camera view.")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class SudokuVideoProcessor(VideoProcessorBase):
        def __init__(self):
            # FIX #1: Reuse the globally cached classifier — NO second model load
            self.clf = global_classifier

            self.last_time = time.time()
            self.fps_buf = []

            self.last_grid_center = None
            self.last_grid_area = None
            self.stabilizing_start_time = None
            self.frame_buffer = []  # list of (blur_score, img)

            self.cached_output = None
            self.status_msg = "SEARCHING GRID"
            self.no_grid_count = 0

            # FIX #3: Frame skip — only run CV detection every N frames
            self.frame_count = 0
            self.DETECT_EVERY_N = 3  # Run grid detection on every 3rd frame

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # FIX #2: All imports are now at module level — no per-frame import cost
            img = frame.to_ndarray(format="bgr24")

            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            fps = 1.0 / dt if dt > 0 else 30.0
            self.fps_buf.append(fps)
            if len(self.fps_buf) > 15:
                self.fps_buf.pop(0)
            avg_fps = sum(self.fps_buf) / len(self.fps_buf)

            # If we have a solved result cached, just stamp and return — zero CV cost
            if self.cached_output is not None:
                display = self.cached_output.copy()
                cv2.putText(display, f"FPS: {avg_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, "STATUS: SOLVED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(display, format="bgr24")

            # FIX #3: Frame skip — skip CV detection on non-sampled frames
            self.frame_count += 1
            if self.frame_count % self.DETECT_EVERY_N != 0:
                display = img.copy()
                cv2.putText(display, f"FPS: {avg_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"STATUS: {self.status_msg}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                return av.VideoFrame.from_ndarray(display, format="bgr24")

            # Image quality checks (runs on sampled frames only)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
            glare_ratio = cv2.countNonZero(glare_mask) / (img.shape[0] * img.shape[1])
            brightness = np.mean(gray)

            board_cnt, _, _ = find_board(img)

            if board_cnt is not None:
                self.status_msg = "DETECTING"
                self.no_grid_count = 0

                M = cv2.moments(board_cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                current_area = cv2.contourArea(board_cnt)

                # Stability Analysis
                if self.last_grid_center is not None and self.last_grid_area is not None:
                    dx = abs(cx - self.last_grid_center[0])
                    dy = abs(cy - self.last_grid_center[1])
                    area_diff = abs(current_area - self.last_grid_area)
                    if area_diff > self.last_grid_area * 0.05 or dx > 15 or dy > 15:
                        self.stabilizing_start_time = time.time()
                        self.frame_buffer = []
                else:
                    self.stabilizing_start_time = time.time()
                    self.frame_buffer = []

                self.last_grid_center = (cx, cy)
                self.last_grid_area = current_area

                elapsed = time.time() - self.stabilizing_start_time
                countdown = max(0, math.ceil(5.0 - elapsed))

                if blur_score < 100 or glare_ratio > 0.02:
                    self.stabilizing_start_time = time.time()
                else:
                    self.status_msg = f"STABILIZING {countdown}"
                    self.frame_buffer.append((blur_score, img.copy()))
                    if len(self.frame_buffer) > 10:
                        self.frame_buffer.pop(0)

                # Multi-Frame Capture: only when stable for 5s
                if elapsed >= 5.0 and len(self.frame_buffer) > 0:
                    self.status_msg = "RUNNING OCR..."

                    # Pick sharpest frame
                    self.frame_buffer.sort(key=lambda x: x[0], reverse=True)
                    best_blur, best_img = self.frame_buffer[0]

                    result = process_multi_frame(self.frame_buffer, self.clf)

                    DEBUG_MODE = os.getenv("DEBUG_MODE", "False") == "True"

                    if result.get("success", False):
                        self.status_msg = "SOLVED"
                        self.cached_output = result["output_frame"]

                        if DEBUG_MODE:
                            os.makedirs("captures", exist_ok=True)
                            original_warped = result["warped"]
                            cv2.imwrite("captures/original.png", original_warped)

                            # Generate side-by-side
                            solved_grid_raw = np.full_like(original_warped, 255)
                            display_numbers_on_board(solved_grid_raw, result["board"], np.zeros(81, dtype=int), new_color=(255, 0, 0))
                            cv2.imwrite("captures/recognized.png", solved_grid_raw)

                            solved_grid = np.full_like(original_warped, 255)
                            display_numbers_on_board(solved_grid, result["solved_board"], result["board"],
                                                     new_color=(0, 200, 0), orig_color=(0, 0, 0))
                            cv2.imwrite("captures/solved.png", solved_grid)

                            final_result = np.hstack((original_warped, solved_grid_raw, solved_grid))
                            cv2.imwrite("captures/final_side_by_side.png", final_result)

                        confs = [c["confidence"] for c in result.get("cell_details", []) if not c["is_empty"]]
                        avg_conf = sum(confs) / len(confs) if confs else 0.0

                        metrics = {
                            "sharpness": best_blur,
                            "glare": glare_ratio,
                            "brightness": brightness,
                            "recognized_count": result.get("recognized_count", 0),
                            "avg_conf": avg_conf,
                            "detected_matrix": result["board"],
                            "solved_matrix": result["solved_board"],
                            "board_validity": "VALID" if not result.get("validation_errors") else "INVALID",
                            "solver_status": result.get("stage", "UNKNOWN")
                        }
                        if DEBUG_MODE:
                            with open("captures/metrics.json", "w") as f:
                                json.dump(metrics, f)
                    else:
                        self.status_msg = "DETECTING"
                        metrics = {
                            "sharpness": best_blur,
                            "glare": glare_ratio,
                            "brightness": brightness,
                            "recognized_count": result.get("recognized_count", 0),
                            "board_validity": "INVALID",
                            "solver_status": result.get("error", "Failed")
                        }
                        if DEBUG_MODE:
                            os.makedirs("captures", exist_ok=True)
                            with open("captures/metrics.json", "w") as f:
                                json.dump(metrics, f)
                        self.stabilizing_start_time = time.time()
                        self.frame_buffer = []

                display = img.copy()
                if self.cached_output is None:
                    cv2.drawContours(display, [board_cnt], -1, (0, 255, 0), 2)
                    cv2.putText(display, f"FPS: {avg_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    color = (0, 255, 255) if "STABILIZING" in self.status_msg else (0, 255, 0)
                    cv2.putText(display, f"STATUS: {self.status_msg}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                return av.VideoFrame.from_ndarray(display, format="bgr24")
            else:
                # Debounce: require 3 consecutive no-grid frames before clearing state
                self.no_grid_count = getattr(self, 'no_grid_count', 0) + 1
                if self.no_grid_count >= 3:
                    self.last_grid_center = None
                    self.last_grid_area = None
                    self.stabilizing_start_time = None
                    self.frame_buffer = []
                    self.cached_output = None
                    self.status_msg = "NO GRID"
                    self.no_grid_count = 0

                display = img.copy()
                cv2.putText(display, f"FPS: {avg_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"STATUS: {self.status_msg}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(display, format="bgr24")

    webrtc_streamer(
        key="sudoku-solver",
        video_processor_factory=SudokuVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        async_processing=True
    )

    st.markdown("---")
    st.subheader("📸 Captured Results")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Refresh View", type="primary"):
            pass

    if os.path.exists("captures/final_side_by_side.png"):
        st.success("✅ Stable Capture Processed!")
        st.image("captures/final_side_by_side.png", caption="Left: Original | Center: Recognized OCR | Right: Final Solved", use_container_width=True)

        if os.path.exists("captures/metrics.json"):
            with open("captures/metrics.json", "r") as f:
                metrics = json.load(f)

            st.markdown("### 🔬 Phase 10 Debug Panel")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sharpness Score", f"{metrics.get('sharpness', 0):.1f}")
            m2.metric("Glare Area", f"{metrics.get('glare', 0)*100:.2f}%")
            m3.metric("Brightness", f"{metrics.get('brightness', 0):.1f}")
            m4.metric("Recognized Digits", metrics.get("recognized_count", 0))

            m5, m6, m7, _ = st.columns(4)
            m5.metric("Avg OCR Conf", f"{metrics.get('avg_conf', 0):.3f}")

            validity = metrics.get('board_validity', 'UNKNOWN')
            m6.metric("Board Validity", validity,
                      delta="Pass" if validity == "VALID" else "Fail",
                      delta_color="normal" if validity == "VALID" else "inverse")

            solver_stat = metrics.get('solver_status', 'UNKNOWN')
            m7.metric("Solver Status", solver_stat)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Detected Matrix (OCR)**")
                if metrics.get("detected_matrix"):
                    st.code(str(np.array(metrics.get("detected_matrix")).reshape(9, 9)))
                else:
                    st.info("No matrix extracted.")
            with c2:
                st.markdown("**Solved Matrix**")
                if metrics.get("solved_matrix"):
                    st.code(str(np.array(metrics.get("solved_matrix")).reshape(9, 9)))
                else:
                    st.error("Solver failed or aborted.")
    else:
        st.info("No captures available. Hold a Sudoku board to the camera for 5 seconds, then click Refresh.")

# ─── Analytics ───────────────────────────────────────────────────────────────
elif page == "Results & Analytics":
    st.title("📊 Model Analytics")
    if not st.session_state.history:
        st.info("No inference data available yet.")
    else:
        st.line_chart([h['processing_time'] for h in st.session_state.history])

# ─── Settings ────────────────────────────────────────────────────────────────
elif page == "Settings":
    st.title("⚙️ Engine Configuration")
    st.slider("Confidence Threshold", 0.1, 1.0, 0.3, step=0.05)
    st.button("Save Configuration", type="primary")
