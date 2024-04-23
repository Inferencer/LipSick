import subprocess
import gradio as gr
import os

def process_files(source_video, driving_audio):
    # Define paths to your pretrained model and output paths
    pretrained_model_path = "./asserts/pretrained_lipsick.pth"
    deepspeech_model_path = "./asserts/output_graph.pb"
    res_video_dir = "./asserts/inference_result"

    # Extract the base name of the source video file without extension
    base_name = os.path.splitext(os.path.basename(source_video.name))[0]
    output_video_name = f"{base_name}_LIPSICK.mp4"  # Updated naming convention
    output_video_path = os.path.join(res_video_dir, output_video_name)

    # Ensure unique filename to prevent overwriting
    output_video_path = get_versioned_filename(output_video_path)

    # Construct command without the landmark path argument
    cmd = [
        'python', 'inference.py',
        '--source_video_path', source_video.name,
        '--driving_audio_path', driving_audio.name,
        '--pretrained_lipsick_path', pretrained_model_path,
        '--deepspeech_model_path', deepspeech_model_path,
        '--res_video_dir', res_video_dir
    ]

    # Run the inference script as a subprocess and handle possible errors
    try:
        subprocess.run(cmd, check=True)
        return output_video_path  # Return the dynamically created file path
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return "An error occurred during processing. Please check the console log."

def get_versioned_filename(filepath):
    """ Append a version number to the filepath if it already exists. """
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}({counter}){ext}"
        counter += 1
    return filepath

iface = gr.Interface(
    fn=process_files,
    inputs=[
        gr.File(label="Upload MP4 File", type="filepath", file_types=["mp4"]),
        gr.File(label="Upload WAV File", type="filepath", file_types=["wav"])
    ],
    outputs=gr.components.Video(label="Processed Video", show_label=False),  # Video component for playback
    title="🤢 LIPSICK 🤮",
    description="Upload your video and driving audio to Lipsync.",
    allow_flagging="never"  # Disable flagging
)

iface.launch()
