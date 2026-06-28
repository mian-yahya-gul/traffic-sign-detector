"""
app.py
Gradio web demo for real-time traffic sign detection and recognition
using a YOLOv8 model fine-tuned via transfer learning.

Run locally:
    python app.py

Deploy:
    Push this repo to a Hugging Face Space (Gradio SDK) — it will
    auto-launch using this file.
"""

import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import time

MODEL_PATH = "best_roboflow.pt"

# ── Load model once at startup ──────────────────────────────────────
print("Loading YOLOv8 traffic sign detection model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")


def detect_signs(image, confidence_threshold=0.4):
    """
    Runs YOLOv8 inference on a single image and returns:
    - the annotated image (with bounding boxes + labels drawn)
    - a structured summary of detections (label, confidence)
    - inference time in milliseconds
    """
    if image is None:
        return None, "No image provided.", ""

    start_time = time.time()
    results = model.predict(image, conf=confidence_threshold, verbose=False)
    inference_time_ms = (time.time() - start_time) * 1000

    result = results[0]
    annotated = result.plot()  # numpy array, BGR
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append((label, conf))

    if not detections:
        summary = "No traffic signs detected above the confidence threshold."
    else:
        detections.sort(key=lambda x: x[1], reverse=True)
        lines = [f"• **{label}** — {conf:.1%} confidence" for label, conf in detections]
        summary = "\n".join(lines)

    timing_info = f"Inference time: {inference_time_ms:.1f} ms | Detections: {len(detections)}"

    return Image.fromarray(annotated_rgb), summary, timing_info


def detect_signs_video(video_path, confidence_threshold=0.4):
    """
    Runs YOLOv8 inference frame-by-frame on an uploaded video and
    returns the path to an annotated output video.
    """
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "annotated_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

    cap.release()
    writer.release()

    return output_path

def detect_signs_webcam(image, confidence_threshold=0.4):
    """
    Runs YOLOv8 inference on a single webcam-captured frame.
    Reuses the same detection logic as image upload.
    """
    annotated, summary, timing = detect_signs(image, confidence_threshold)
    return annotated, summary, timing


# ── Gradio UI ────────────────────────────────────────────────────────
with gr.Blocks(title="Traffic Sign Detection & Recognition") as demo:
    gr.Markdown(
        """
        # 🚦 Automated Traffic Sign Detection & Recognition
        Real time traffic sign detection using a YOLOv8 model fine tuned via transfer learning.
        Upload an image or video to see detected traffic signs, their classification, and confidence scores.

        **Use case:** driver assistance systems, dashcam footage auditing for road safety compliance,
        and as a building block for autonomous vehicle perception pipelines.
        """
    )

    with gr.Tab("Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload a traffic sign image")
                confidence_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                    label="Confidence Threshold"
                )
                image_button = gr.Button("Detect Signs", variant="primary")
            with gr.Column():
                image_output = gr.Image(type="pil", label="Detection Result")
                detection_summary = gr.Markdown(label="Detected Signs")
                timing_output = gr.Textbox(label="Performance", interactive=False)

        image_button.click(
            fn=detect_signs,
            inputs=[image_input, confidence_slider],
            outputs=[image_output, detection_summary, timing_output],
        )

        gr.Examples(
            examples=[],  # add paths to example images here, e.g. "examples/stop_sign.jpg"
            inputs=image_input,
            label="Try an example",
        )

    with gr.Tab("Video Detection"):
        gr.Markdown("Upload a short driving/dashcam clip to see frame-by-frame traffic sign detection.")
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload a video")
                video_confidence_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                    label="Confidence Threshold"
                )
                video_button = gr.Button("Process Video", variant="primary")
            with gr.Column():
                video_output = gr.Video(label="Annotated Output")

        video_button.click(
            fn=detect_signs_video,
            inputs=[video_input, video_confidence_slider],
            outputs=video_output,
        )
    with gr.Tab("Live Webcam"):
        gr.Markdown(
            "Capture a frame from your webcam to test detection in real time. "
            "Click the camera icon to take a snapshot, then press **Detect Signs**."
        )
        with gr.Row():
            with gr.Column():
                webcam_input = gr.Image(sources=["webcam"], type="pil", label="Webcam Capture")
                webcam_confidence_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                    label="Confidence Threshold"
                )
                webcam_button = gr.Button("Detect Signs", variant="primary")
            with gr.Column():
                webcam_output = gr.Image(type="pil", label="Detection Result")
                webcam_summary = gr.Markdown(label="Detected Signs")
                webcam_timing = gr.Textbox(label="Performance", interactive=False)

        webcam_button.click(
            fn=detect_signs_webcam,
            inputs=[webcam_input, webcam_confidence_slider],
            outputs=[webcam_output, webcam_summary, webcam_timing],
        )

    gr.Markdown(
        """
        ---
        Built by [Mian Yahya Gul](https://github.com/mian-yahya-gul) ·
        [LinkedIn](https://www.linkedin.com/in/mian-yahya-gul/) ·
        Model trained with YOLOv8 via transfer learning.
        """
    )

if __name__ == "__main__":
    demo.launch()
