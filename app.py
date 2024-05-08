import subprocess
import gradio as gr
import os
from compute_crop_radius import calculate_crop_radius_statistics

def compute_crop_radius_stats(video_file):
    lowest, highest, average, most_common = calculate_crop_radius_statistics(video_file.name)
    return f"Lowest: {lowest}, Highest: {highest}, Average: {average}, Most Common: {most_common}", lowest, highest, average, most_common

def process_files(source_video, driving_audio, custom_crop_radius=None):
    pretrained_model_path = "./asserts/pretrained_lipsick.pth"
    deepspeech_model_path = "./asserts/output_graph.pb"
    res_video_dir = "./asserts/inference_result"

    base_name = os.path.splitext(os.path.basename(source_video))[0]
    output_video_name = f"{base_name}_LIPSICK.mp4"
    output_video_path = os.path.join(res_video_dir, output_video_name)

    output_video_path = get_versioned_filename(output_video_path)

    cmd = [
        'python', 'inference.py',
        '--source_video_path', source_video,
        '--driving_audio_path', driving_audio,
        '--pretrained_lipsick_path', pretrained_model_path,
        '--deepspeech_model_path', deepspeech_model_path,
        '--res_video_dir', res_video_dir,
    ]

    if custom_crop_radius is not None:
        cmd.extend(['--custom_crop_radius', str(custom_crop_radius)])

    try:
        subprocess.run(cmd, check=True)
        return output_video_path
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return "An error occurred during processing. Please check the console log."


def get_versioned_filename(filepath):
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}({counter}){ext}"
        counter += 1
    return filepath

with gr.Blocks() as iface:
    gr.Markdown("### 🤢 LIPSICK 🤮\nUpload your video and driving audio to Lipsync.")
    
    with gr.Tab("Process Video"):
        source_video = gr.File(label="Upload MP4 File", type="filepath", file_types=["mp4"])
        driving_audio = gr.File(label="Upload Audio File", type="filepath", file_types=["mp3", "wav", "aac", "wma", "flac", "m4a"])
        custom_crop_radius = gr.Number(label="Custom Crop Radius (Optional) - See Compute Crop Radius Stats tab to calculate this value ", value=None)
        process_btn = gr.Button("Process Video")
        output_video = gr.Video(label="Processed Video", show_label=False)
        
        process_btn.click(
            process_files,
            inputs=[source_video, driving_audio, custom_crop_radius],
            outputs=[output_video]
        )
    
    with gr.Tab("Compute Crop Radius Stats"):
        video_file = gr.File(label="Upload MP4 File", type="filepath", file_types=["mp4"])
        compute_btn = gr.Button("Compute Crop Radius Stats")
        stats_output = gr.Textbox(label="Crop Radius Stats")
        lowest = gr.Number(label="Lowest Crop Radius (Not recommended)", interactive=False)
        highest = gr.Number(label="Highest Crop Radius(Not recommended)", interactive=False)
        average = gr.Number(label="Average Crop Radius (Recommended Value)", interactive=False)
        most_common = gr.Number(label="Most Common Crop Radius(Recommended Value)", interactive=False)
        
        compute_btn.click(
            compute_crop_radius_stats,
            inputs=[video_file],
            outputs=[stats_output, lowest, highest, average, most_common]
        )
        
iface.launch(inbrowser=True)
