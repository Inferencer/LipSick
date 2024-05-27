
<a href="https://youtu.be/ZO2AaXXwMrw">
  <img src="/utils/logo/LipSickYouTube.jpg" alt="LipSick Logo">
</a>


## Introduction

#### To get started with LipSick on Windows, follow these steps to set up your environment. This branch has been tested with Anaconda using Python 3.10 and CUDA 11.6 & CUDA 11.8 with only 4GB VRAM. Using a different Cuda version can cause speed issues.
See branches for [Linux](https://github.com/Inferencer/LipSick/tree/linux) or HuggingFace [GPU](https://github.com/Inferencer/LipSick/tree/HuggingFace-GPU) / [CPU](https://github.com/Inferencer/LipSick/tree/HuggingFace-CPU) or [Collab](https://github.com/Inferencer/LipSick/tree/Google-Collab)

## Setup

<details>
  <summary>Install</summary>

1. Clone the repository:
```bash
git clone https://github.com/Inferencer/LipSick.git
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
Or use the new autorun tool by double clicking run_lipsick.bat


This will launch a Gradio interface where you can upload your video and audio files to process them with LipSick.





### To-Do List

- [ ] Add support MacOS.
- [ ] Add upscale reference frames with masking. 
- [ ] Add alternative option for face tracking model [SFD](https://github.com/1adrianb/face-alignment) (likely best results, but slower than Dlib).
- [ ] Examine CPU speed upgrades.
- [ ] Reintroduce persistent folders for frame extraction as an option with existing frame checks for faster extraction on commonly used videos. 😷
- [ ] Provide HuggingFace space CPU (free usage but slower). 😷
- [ ] Release Tutorial on manual masking using DaVinci. 😷
- [ ] Image to MP4 conversion so a single image can be used as input.
- [ ] Automatic audio conversion to WAV regardless of input audio format. 🤕
- [ ] Clean README.md & provide command line inference.
- [ ] Remove input video 25fps requirement.
- [ ] Upload cherry picked input footage for user download & use.
- [ ] Create a Discord to share results, faster help, suggestions & cherry picked input footage.
- [ ] Multi face Lipsync on large scene scene changes/ cut scenes
- [ ] Mutli face Lipsync support on 1+ person in video.
- [ ] skipable frames when no face it detected.
- [ ] Close mouth fully on silence
- [x] Add visualization for custom ref frames & print correct values 🤮
- [x] Add auto masking to remove the common bounding box around mouths. 🤢
- [x] Provide Google Colab .IPYNB. 🤮
- [x] Add support for Linux. 🤢
- [x] Looped original video generated as an option for faster manual masking. 🤮
- [x] Upload results footage montage to GitHub so new users can see what LipSick is capable of. 🤮
- [x] Add custom reference frame feature. 🤮
- [x] auto git pull updater .bat file 🤢
- [x] Add auto persistent crop_radius to prevent mask flickering. 🤮
- [x] Auto run the UI with a .bat file. 🤮
- [x] Auto open UI in default browser. 🤮
- [x] Add custom crop radius feature to stop flickering [Example](https://github.com/Inferencer/LipSick/issues/8#issuecomment-2099371266) 🤮
- [x] Provide HuggingFace space GPU. 🤮
- [x] Remove warning messages in command prompt that don't affect performance. 🤢
- [x] Moved frame extraction to temp folders. 🤮
- [x] Results with the same input video name no longer overwrite existing results. 🤮
- [x] Remove OpenFace CSV requirement. 🤮
- [x] Detect accepted media input formats only. 🤮
- [x] Upgrade to Python 3.10. 🤮
- [x] Add UI. 🤮

### Key:
- 🤮 = Completed & published
- 🤢 = Completed & published but requires community testing
- 😷 = Tested & working but not published yet
- 🤕 = Tested but not ready for public use
### Simple Key:
- [x] Available
- [ ] Unavailable

## Acknowledge

This project, LipSick, is heavily inspired by and based on [DINet](https://github.com/MRzzm/DINet). Specific components are borrowed and adapted to enhance LipSick


We express our gratitude to the authors and contributors of DINet for their open-source code and documentation.
