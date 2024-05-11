![LipSick Logo](/utils/logo/LipSick_bg.jpg)

## Introduction

#### To get started with LipSick on Linux, follow these steps to set up your environment. This branch has been tested with HuggingFace spaces which runs Linux using Python 3.10 and CUDA 12.3.2.

## Setup

<details>
  <summary>Install</summary>

1. Clone the repository:
```bash
git clone -b linux https://github.com/Inferencer/LipSick.git
cd LipSick
```
2. Create and activate the Anaconda environment:
```bash
conda env create -f environment.yml
conda activate LipSick
```
</details>

## Download pre-trained models
<details>
  <summary>Download Links</summary>

### For the folder ./asserts

Please download pretrained_lipsick.pth using this [link](https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/pretrained_lipsick.pth) and place the file in the folder ./asserts

Then, download output_graph.pb using this [link](https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/output_graph.pb) and place the file in the same folder.

### For the folder ./models

Please download shape_predictor_68_face_landmarks.dat using this [link](https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/shape_predictor_68_face_landmarks.dat) and place the file in the folder ./models
</details>

### The folder structure for manually downloaded models
```bash
.
├── ...
├── asserts                        
│   ├── examples                   # A place to store inputs if not using gradio UI
│   ├── inference_result           # Results will be saved to this folder
│   ├── output_graph.pb            # The DeepSpeech model you manually download and place here
│   └── pretrained_lipsick.pth     # Pre-trained model you manually download and place here
│                   
├── models
│   ├── Discriminator.py
│   ├── LipSick.py
│   ├── shape_predictor_68_face_landmarks.dat  # Dlib Landmark tracking model you manually download and place here
│   ├── Syncnet.py
│   └── VGG19.py   
└── ...
```
4. Run the application:
```bash
python app.py
```


This will launch a Gradio interface where you can upload your video and audio files to process them with LipSick.
Changelog




## Acknowledge

This project, LipSick, is heavily inspired by and based on [DINet](https://github.com/MRzzm/DINet). Specific components are borrowed and adapted to enhance LipSick


We express our gratitude to the authors and contributors of DINet for their open-source code and documentation.
