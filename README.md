![LipSick Logo](/utils/logo/LipSick.png)

## Installation Instructions

### To get started with LipSick on Windows, follow these steps to set up your environment. This branch has been tested with Anaconda using Python 3.10 and CUDA 11.6.


1. Clone the repository:
```python
git clone https://github.com/Inferencer/LipSick.git
cd LipSick
```
2. Create and activate the Anaconda environment:
```python
conda env create -f environment.yml
conda activate lipsick
```
## Download
### For the folder ./asserts

Please download pretrained_lipsick.pth using this [link](https://github.com/Inferencer/LipSick/releases/download/v1pretrained_lipsick.pth/pretrained_lipsick.pth) and place the file in the folder ./asserts

Then, download output_graph.pb using this [link](https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/output_graph.pb) and place the file in the same folder.

### For the folder ./models

Please download shape_predictor_68_face_landmarks.dat using this [link](https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/shape_predictor_68_face_landmarks.dat) and place the file in the folder ./models

### The folder structure for manually donwloaded models
```python
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
```python
python app.py
```


This will launch a Gradio interface where you can upload your video and audio files to process them with LipSick.
Changelog





 Add support for Linux and MacOS.
 Integrate with Hugging Face and Google Colab.
 Optimize model performance for different hardware setups.

### To-Do List

- [ ] Add support for Linux and MacOS.
- [ ] Integrate with Hugging Face and Google Colab.
- [ ] Add upscale reference frames with masking.
- [ ] Add seamless clone masking to remove the common bounding box around mouths.
- [ ] Add custom reference frame feature.
- [ ] Add auto persistent crop_radius to prevent mask flickering.
- [ ] Examine CPU speed upgrades.
- [ ] Reintroduce persistent folders for frame extraction as an option with existing frame checks for faster extraction on commonly used videos.
- [ ] Provide HuggingFace space CPU (free usage but slower).
- [ ] Provide Google Colab .IPYNB.
- [ ] Provide HuggingFace space GPU.
- [ ] Release Tutorial on manual masking using DaVinci.
- [ ] Looped original video generated as an option for faster manual masking.
- [ ] Image to MP4 conversion so a single image can be used as input.
- [ ] Automatic audio conversion to WAV regardless of input audio format.
- [x] Moved frame extraction to temp folders.
- [x] Results with the same input video name no longer overwrite existing results.
- [x] Remove OpenFace CSV requirement.
- [x] Detect accepted media input formats only.
- [x] Upgrade to Python 3.10.
- [x] Add UI.


## Acknowledge

This project, LipSick, is heavily inspired by and based on [DINet](https://github.com/MRzzm/DINet). Specific components are borrowed and adapted to enhance LipSick


We express our gratitude to the authors and contributors of DINet for their open-source code and documentation.
