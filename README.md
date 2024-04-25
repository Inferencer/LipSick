<div align="center">
    <h1>🤢 LipSick Collab 🤮</h1> 
    <h1><a href="https://colab.research.google.com/drive/172-G93PTzioP67DoltOliMOn5vnBSeRW?usp=sharing">Google Collab Link</a></h1>
</div>


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
- [x] Provide Google Colab .IPYNB. 🤮
- [x] Add support for Linux. 🤢
- [ ] Release Tutorial on manual masking using DaVinci. 😷
- [ ] Looped original video generated as an option for faster manual masking. 😷
- [ ] Image to MP4 conversion so a single image can be used as input.
- [x] Automatic audio conversion to WAV regardless of input audio format. 🤮
- [ ] Clean README.md & provide command line inference.
- [ ] Remove input video 25fps requirement.
- [ ] Upload cherry picked input footage for user download & use.
- [ ] Create a Discord to share results, faster help, suggestions & cherry picked input footage.
- [ ] Upload results footage montage to GitHub so new users can see what LipSick is capable of.
- [x] Provide HuggingFace space GPU. 🤮
- [x] Remove warning messages in command prompt that don't affect performance. 🤢
- [x] Moved frame extraction to temp folders. 🤮
- [x] Results with the same input video name no longer overwrite existing results. 🤮
- [x] Remove OpenFace CSV requirement. 🤮
- [x] Detect accepted media input formats only. 🤮
- [x] Upgrade to Python 3.10. 🤮
- [x] Add UI. 🤮

    <img src="https://github.com/Inferencer/LipSick/blob/main/utils/logo/LipSick_bg.jpg" alt="LipSick Logo" style="max-width:100%;">

## Acknowledge

This project, LipSick, is heavily inspired by and based on [DINet](https://github.com/MRzzm/DINet). Specific components are borrowed and adapted to enhance LipSick
