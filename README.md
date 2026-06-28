---
title: Traffic Sign Detection
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# 🚦 Automated Traffic Sign Detection & Recognition

A real-time traffic sign detection and classification system built with **YOLOv8** via transfer learning, deployed as an interactive web demo — no installation required to try it.

**🔗 [Live Demo](#)** ← add your Hugging Face Space URL here after deploying

## Use Case

Traffic sign recognition is a foundational perception task for:
- **Driver-assistance systems (ADAS)** — alerting drivers to speed limits, stop signs, and warnings in real time
- **Dashcam footage auditing** — automatically flagging missed or obscured signage for road-safety compliance reviews
- **Autonomous vehicle perception pipelines** — as a building-block detection module feeding into higher-level driving decisions

This project demonstrates the full pipeline from a trained model to a usable, public-facing tool — not just a notebook.

## How it works

1. A YOLOv8 model was fine-tuned via transfer learning on a traffic sign dataset
2. The trained weights (`best_roboflow.pt`) power a Gradio web interface
3. Users upload an image or video; the app runs inference and returns annotated output with bounding boxes, class labels, and confidence scores
4. Both single-image and frame-by-frame video processing are supported

## Tech Stack

`Python` · `YOLOv8 (Ultralytics)` · `Gradio` · `OpenCV` · `Transfer Learning`

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

> **Note:** you'll need the trained model file `best_roboflow.pt` in the project root — it's not committed to this repo due to file size; see [Model Weights](#model-weights) below.

## Model Weights

The trained YOLOv8 weights are hosted separately due to file size:
- **Hugging Face Spaces deployment:** the model is uploaded directly to the Space's file storage
- **Local use:** download the model from [link to be added] and place it in the project root as `best_roboflow.pt`

## Deploying your own copy

This repo is structured to deploy directly to **Hugging Face Spaces**:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), choosing the **Gradio** SDK
2. Push this repo's contents to the Space's git remote
3. Upload `best_roboflow.pt` to the Space's file browser (or via `git lfs` if using git push)
4. The Space will auto-build and launch `app.py`

## Project Structure

```
.
├── app.py              # Gradio web app (image + video detection)
├── requirements.txt
├── best_roboflow.pt     # Trained YOLOv8 weights (not committed — see Model Weights)
└── examples/            # Sample images for the demo's "Try an example" section
```

## Author

**Mian Yahya Gul** — [LinkedIn](https://www.linkedin.com/in/mian-yahya-gul/) · [GitHub](https://github.com/mian-yahya-gul)
