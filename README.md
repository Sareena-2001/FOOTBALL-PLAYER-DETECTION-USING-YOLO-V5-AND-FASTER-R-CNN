# Football Player Detection Using YOLOv5 and Faster R-CNN

A computer-vision project exploring football-player detection in match footage with two object-detection approaches: **YOLOv5** and **Faster R-CNN with a ResNet-50 FPN backbone**. The repository contains Python scripts, Google Colab notebooks, a sample match video, and several rendered output frames.

> **Important:** The repository demonstrates object detection on video frames, but it does not include a complete, self-contained custom training dataset or a `data.yaml` file. The included scripts also have Colab-specific paths and setup commands, so they need adjustment before they can run as ordinary local Python scripts.

## Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Models](#models)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Run in Google Colab](#run-in-google-colab)
- [YOLOv5 Workflow](#yolov5-workflow)
- [Faster R-CNN Workflow](#faster-r-cnn-workflow)
- [Outputs](#outputs)
- [Implementation Notes and Limitations](#implementation-notes-and-limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview

The project processes football match video frame by frame and applies object-detection models to identify objects in the frames. The two approaches included are:

- **YOLOv5:** Loads a YOLOv5 model through `torch.hub`, processes video frames, filters displayed detections by confidence, and renders detections for inspection. The notebook also contains a command intended to train YOLOv5 on a custom dataset.
- **Faster R-CNN:** Loads a pretrained Faster R-CNN model with a ResNet-50 FPN backbone from `torchvision`, processes up to six frames, draws bounding boxes for detections above a confidence threshold, saves the annotated frames, and displays them.

The included scripts do not implement player tracking across frames, team classification, jersey-number recognition, match statistics, or a deployed application.

## Features

- Video-frame reading with OpenCV.
- Object detection with YOLOv5 and Faster R-CNN.
- Confidence-threshold filtering.
- Bounding-box rendering on frames.
- Display of processed frames in a notebook environment.
- Saving of annotated frames in the Faster R-CNN workflow.
- A sample video and sample output images are included in the repository.

## Models

### YOLOv5

The YOLO workflow clones the Ultralytics YOLOv5 repository and installs its requirements in a Colab notebook. It includes a training command configured for 100 epochs, an image size of 640, a batch size of 16, and the `yolov5x.pt` weights. The training command expects a dataset configuration file at:

```text
/content/drive/MyDrive/football_players_detection/data.yaml
```

That `data.yaml` file and the referenced custom dataset are not included in this repository archive.

The later inference section loads `yolov5x` through `torch.hub.load('ultralytics/yolov5', 'yolov5x')`. It does not explicitly load the `custom_model` weights produced by the preceding training command. Therefore, the included inference code should not be described as using the trained custom model unless it is updated to load the appropriate weights.

### Faster R-CNN

The Faster R-CNN script uses:

```python
fasterrcnn_resnet50_fpn(pretrained=True)
```

This loads a pretrained torchvision model. The script sets the model to evaluation mode, processes video frames, and filters detections using a confidence threshold of `0.9`.

The script labels every displayed detection as `Person`, regardless of the model's predicted class ID. The pretrained detector is not fine-tuned on a football-player dataset in the included code.

## Repository Structure

```text
.
├── OUTPUT IMAGES/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── frame_0003.png
├── USING FASTER R-CNN/
│   ├── TASK_3_DETECTING_FOOTBALL_PLAYERS_USING_FASTER_R_CNN.ipynb
│   └── task_3_detecting_football_players_using_faster_r_cnn.py
├── USING YOLO/
│   ├── TASK_3_DETECTING_FOOTBALL_PLAYERS_USING_YOLO_V5.ipynb
│   └── task_3_detecting_football_players_using_yolo_v5.py
└── VIDEO/
    └── match.mp4
```

The `.ipynb` files are notebook versions of the workflows. The `.py` files were exported from Google Colab and retain Colab-specific code and notebook shell-command syntax in places.

## Requirements

The code imports or uses:

- Python
- PyTorch
- `torchvision`
- OpenCV (`cv2`)
- NumPy
- Pillow
- IPython display utilities
- Ultralytics YOLOv5 and its dependencies
- Google Colab and Google Drive for the original notebook paths

No `requirements.txt`, pinned environment file, custom training dataset, or `data.yaml` file was included in the inspected repository. Exact tested package versions therefore cannot be confirmed.

## Run in Google Colab

The notebooks were written for Google Colab and contain Google Drive mounting code.

1. Open the relevant notebook in Google Colab.
2. Upload the sample video to your Google Drive, or update the video path to point to a file you have access to.
3. Run the Drive-mount cell and authorize access.
4. Update the `video_path` and output directory paths to match your Drive folders.
5. Run the remaining cells in order.

The Faster R-CNN notebook expects a video at:

```text
/content/drive/MyDrive/football_players_detection/video.mp4
```

The YOLO notebook expects:

```text
/content/drive/MyDrive/football_players_detection/match.mp4
```

These are separate hard-coded paths. Update them to the actual location and filename of the video you intend to process.

### YOLO training setup

To run the YOLO training cell, first prepare a dataset in the format expected by YOLOv5 and create a valid `data.yaml` file. Place it at the configured path or update the command. The training cell also clones the YOLOv5 repository and installs its dependencies.

The dataset configuration and training images are not supplied in this repository, so the custom-training step cannot run as-is.

## YOLOv5 Workflow

The notebook's intended sequence is:

1. Mount Google Drive.
2. Clone the YOLOv5 repository and install its requirements.
3. Run a custom-training command using the configured `data.yaml`.
4. Load a YOLOv5 model through PyTorch Hub.
5. Read the video frame by frame using OpenCV.
6. Run inference on each frame.
7. Filter the detection table by confidence.
8. Render and display annotated frames.

The configured confidence threshold is `0.9`. Detections below this threshold are excluded from the printed filtered-detection table. The code calls `results.render()` to render the model's detections for display.

### Important note about the Python file

`task_3_detecting_football_players_using_yolo_v5.py` contains lines beginning with `!` and notebook-style commands. These are IPython/Colab shell syntax, not valid standard Python syntax. Run the notebook in Colab or convert those commands to terminal commands before running the file with ordinary Python.

## Faster R-CNN Workflow

The Faster R-CNN script:

1. Mounts Google Drive.
2. Creates the configured output directory.
3. Loads the pretrained Faster R-CNN ResNet-50 FPN model.
4. Opens the configured video with OpenCV.
5. Processes up to six frames.
6. Converts each frame to a tensor and runs inference without gradient calculation.
7. Keeps detections with scores of at least `0.9`.
8. Draws bounding boxes and confidence scores on the frame.
9. Saves each annotated frame as a PNG and displays it.

The output directory is configured as:

```text
/content/drive/MyDrive/football_players_detection/outputs
```

The maximum number of frames is controlled by `max_images = 6`. The confidence threshold is controlled by `confidence_threshold = 0.9`.

### Running outside Colab

The Faster R-CNN Python file imports `google.colab`, so it will not run unchanged in a normal local Python environment. Remove the Drive-mount lines and update the video and output paths to local paths. Install compatible versions of PyTorch and torchvision, along with OpenCV, NumPy, and Pillow.

## Outputs

The repository includes four sample PNG images in `OUTPUT IMAGES/`. The Faster R-CNN script is written to save annotated frames as `frame_0000.png`, `frame_0001.png`, and so on, in its configured output directory.

The YOLO notebook displays rendered frames. The included code does not save a complete annotated output video.

The sample output images demonstrate the format of frame-level output; they do not establish model accuracy, recall, precision, or real-world tracking performance.

## Implementation Notes and Limitations

- **Custom training data is missing:** The YOLO training command refers to a `data.yaml` file and dataset that are not included.
- **Custom weights are not used by the later YOLO inference code:** The inference section loads the standard `yolov5x` model through PyTorch Hub rather than explicitly loading the trained `custom_model` weights.
- **Faster R-CNN is pretrained, not football-specific fine-tuned:** The included script uses a pretrained torchvision model and does not train it on this project's own labeled football data.
- **Class labels are simplified:** The Faster R-CNN script displays `Person` for every retained detection rather than checking the model's predicted class.
- **Colab-only paths:** Video and output locations are hard-coded to Google Drive.
- **Python scripts contain notebook syntax:** The YOLO `.py` export includes `!` shell commands, so it is not directly executable as a standard Python script.
- **Frame-level detection only:** The project does not associate detections across frames or maintain player identities.
- **No quantitative evaluation:** The repository does not provide a test dataset or evaluation metrics such as precision, recall, mAP, or FPS.
- **Confidence threshold is fixed:** The threshold of `0.9` may omit valid detections; its suitability has not been evaluated.
- **Video output is not assembled:** The scripts display frames, and the Faster R-CNN workflow saves PNGs, but neither included workflow writes a new annotated video.
- **No dependency lockfile:** Exact package versions and reproducible installation are not documented.

## Future Improvements

1. Add a documented dataset and `data.yaml` file for reproducible YOLO training.
2. Explicitly load the trained YOLO weights during inference.
3. Fine-tune and evaluate the models on labeled football footage.
4. Use model class IDs to label detections correctly.
5. Add precision, recall, mAP, and inference-speed evaluation.
6. Save an annotated video with the original frame rate and dimensions.
7. Add multi-object tracking to follow players across frames.
8. Make input video, output directory, model weights, and confidence threshold configurable.
9. Add robust handling for missing video files and failed frame reads.
10. Provide a `requirements.txt` or environment file with tested package versions.
11. Convert the Colab-exported scripts into clean, directly executable Python programs.
12. Compare YOLOv5 and Faster R-CNN on the same held-out footage and under the same evaluation protocol.

## License

No license file was included in the inspected repository. Unless a license is added, permissions to reuse, modify, and redistribute the code may be restricted. Check the licenses of the external YOLOv5 repository, pretrained model weights, and dependencies separately before redistribution or commercial use.
