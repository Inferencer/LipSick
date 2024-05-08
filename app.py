import subprocess
import gradio as gr
import os
from compute_crop_radius import calculate_crop_radius_statistics

def compute_crop_radius_stats(video_file):
    if video_file is None:
        return "Error: No Video Detected - Upload Video First", None
    lowest, highest, average, most_common = calculate_crop_radius_statistics(video_file.name)
    stats_text = (f"If you want automatic crop_radius then leave the field as 0 or leave it empty. "
                  f"After tracking this video we have found the following info<br><br>"
                  f"Lowest Crop Radius: {lowest}<br>"
                  f"Highest Crop Radius: {highest}<br>"
                  f"Average Crop Radius: {average}<br>"
                  f"Most Common Crop Radius: {most_common}<br>"
                  "_The Most Common Crop Radius is recommended so will be automatically added to the custom crop field below._")
    return stats_text, most_common

def process_files(source_video, driving_audio, custom_crop_radius=None):
    # Insert your existing video processing logic here
    pass  # Placeholder, replace with actual processing logic

with gr.Blocks() as iface:
    gr.Markdown("### 🤢 LIPSICK 🤮\nUpload your video and driving audio to Lipsync.")
    
    with gr.Tab("Process Video"):
        source_video = gr.File(label="Upload MP4 File", type="filepath", file_types=["mp4"])
        driving_audio = gr.File(label="Upload Audio File", type="filepath", file_types=["mp3", "wav", "aac", "wma", "flac", "m4a"])
        
        # Advanced settings dropdown, minimized by default
        with gr.Accordion("Advanced Options", open=False):
            compute_crop_btn = gr.Button("Compute Crop Radius")
            stats_output = gr.Markdown()  # Using Markdown to display styled text
            custom_crop_radius = gr.Number(label="Custom Crop Radius (Optional)", value=None)
            
            compute_crop_btn.click(
                fn=compute_crop_radius_stats,
                inputs=[source_video],
                outputs=[stats_output, custom_crop_radius]
            )

        process_btn = gr.Button("Process Video")
        output_video = gr.Video(label="Processed Video", show_label=False)
        
        process_btn.click(
            fn=process_files,
            inputs=[source_video, driving_audio, custom_crop_radius],
            outputs=[output_video]
        )

iface.launch(inbrowser=True)
