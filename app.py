import subprocess
import gradio as gr
import os
import warnings
from compute_crop_radius import calculate_crop_radius_statistics

# Suppress specific warnings about video conversion
warnings.filterwarnings("ignore", message="Video does not have browser-compatible container or codec. Converting to mp4")

def get_versioned_filename(filepath):
    base, ext = os.path.splitext(filepath)
    counter = 1
    versioned_filepath = filepath
    while os.path.exists(versioned_filepath):
        versioned_filepath = f"{base}({counter}){ext}"
        counter += 1
    return versioned_filepath

def compute_crop_radius_stats(video_file):
    if video_file is None:
        return "Please upload a video file first."
    print("Computing Crop Radius")
    _, _, _, most_common = calculate_crop_radius_statistics(video_file.name)
    print(f"Done: Crop radius = {most_common}")
    return most_common

def process_files(source_video, driving_audio, custom_crop_radius=None, generate_same_length_video=False):
    if not driving_audio:
        print("Please upload audio first.")
        return "", "Error: Audio file is required.", None

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
    if generate_same_length_video:
        cmd.append('--generate_same_length_video')

    try:
        subprocess.run(cmd, check=True)  # Display output from the Python script
        print("Creating low resolution browser playable video")
        print("Complete")
        print("High quality version saved in ./asserts/inference_results")
        print("Thank you for using LipSick for your Lip-Sync needs.")
        
        # Determine the path of the same-length extra video
        same_length_video_path = output_video_path.replace('_facial_dubbing.mp4', '_samelength.mp4')
        
        # Check if generate_same_length_video is checked before returning same_length_video_path
        if generate_same_length_video:
            return "", output_video_path, same_length_video_path
        else:
            return "", output_video_path
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return "An error occurred during processing. Please check the console log.", "", None


with gr.Blocks() as iface:
    gr.Markdown("### 🤢 LIPSICK 🤮\nUpload your video and driving audio to Lipsync.")

    with gr.Tab("Process Video"):
        source_video = gr.File(label="Upload MP4 File", type="filepath", file_types=["mp4"])
        driving_audio = gr.File(label="Upload Audio File", type="filepath", file_types=["mp3", "wav", "aac", "wma", "flac", "m4a"])
        
        with gr.Accordion("Advanced Options", open=False):
            generate_same_length_video = gr.Checkbox(label="Generate same-length extra video without lipsyncing for masking - this be be saved in the folder ./asserts/inference_results", value=False)
            face_tracker_options = ['dlib', 'opencv', 'openface', 'sfd', 'blazeface']
            face_tracker = gr.CheckboxGroup(
                label="Face Tracker",
                choices=face_tracker_options,
                value=['dlib'],
                type="index",
                interactive=False
            )
            gr.Markdown("___")  # Visual separator for layout
            custom_crop_radius = gr.Number(label="Custom Crop Radius (Optional)", value=0)
            compute_crop_btn = gr.Button("Compute Crop Radius Stats")
            compute_crop_btn.click(
                fn=compute_crop_radius_stats,
                inputs=[source_video],
                outputs=[custom_crop_radius]
            )
        gr.Markdown("___")  # Visual spacer

        process_btn = gr.Button("Process Video")
        output_video = gr.Video(label="Processed Video", show_label=False)
        status_text = gr.Textbox(visible=False)  # Reintroduce a hidden Textbox for stability

        process_btn.click(
            fn=process_files,
            inputs=[source_video, driving_audio, custom_crop_radius, generate_same_length_video],
            outputs=[status_text, output_video]
        )

    iface.launch(inbrowser=True)
