---
title: LIPSICK
emoji: 🤮
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
---
I will link to my space when completed so you can just duplicate the space and select the paid hardware Nvidia T4 small is 
the smallest GPU HuggingFace offers which is $0.40/hour charging each minute of uptime regardless of GPU usage unless you pause the space.

I will also upload a HuggingFace CPU branch that will be free but is much slower 🤮
![LipSick Logo](/utils/logo/HuggingFacePrices.jpg)
![LipSick Logo](/utils/logo/LipSick_bg.jpg)

### To-Do List

- [ ] Add support MacOS.
- [ ] Add upscale reference frames with masking. 
- [ ] Add seamless clone masking to remove the common bounding box around mouths. 🤕
- [ ] Add alternative option for face tracking model [SFD](https://github.com/1adrianb/face-alignment) (likely best results, but slower than Dlib).
- [ ] Add custom reference frame feature. 😷
- [ ] Add auto persistent crop_radius to prevent mask flickering. 😷
- [ ] Examine CPU speed upgrades.
- [ ] Reintroduce persistent folders for frame extraction as an option with existing frame checks for faster extraction on commonly used videos. 😷
- [ ] Provide HuggingFace space CPU (free usage but slower). 😷
- [ ] Provide Google Colab .IPYNB. 😷
- [ ] Provide HuggingFace space GPU. 😷
- [ ] Add support for Linux. 🤢
- [ ] Release Tutorial on manual masking using DaVinci. 😷
- [ ] Looped original video generated as an option for faster manual masking. 😷
- [ ] Image to MP4 conversion so a single image can be used as input.
- [ ] Automatic audio conversion to WAV regardless of input audio format.
- [ ] Clean README.md & provide command line inference.
- [ ] Remove input video 25fps requirement.
- [ ] Upload cherry picked input footage for user download & use.
- [ ] Create a Discord to share results, faster help, suggestions & cherry picked input footage.
- [ ] Upload results footage montage to GitHub so new users can see what LipSick is capable of.
- [x] Remove warning messages in command prompt that don't affect performance. 🤢
- [x] Moved frame extraction to temp folders. 🤮
- [x] Results with the same input video name no longer overwrite existing results. 🤮
- [x] Remove OpenFace CSV requirement. 🤮
- [x] Detect accepted media input formats only. 🤮
- [x] Upgrade to Python 3.10. 🤮
- [x] Add UI. 🤮

## Acknowledge

This project, LipSick, is heavily inspired by and based on [DINet](https://github.com/MRzzm/DINet). Specific components are borrowed and adapted to enhance LipSick
