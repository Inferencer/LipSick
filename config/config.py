import argparse

class DataProcessingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument('--extract_video_frame', action='store_true', help='extract video frame')
        self.parser.add_argument('--extract_audio', action='store_true', help='extract audio files from videos')
        self.parser.add_argument('--extract_deep_speech', action='store_true', help='extract deep speech features')
        self.parser.add_argument('--crop_face', action='store_true', help='crop face')
        self.parser.add_argument('--generate_training_json', action='store_true', help='generate training json file')

        self.parser.add_argument('--source_video_dir', type=str, default="./asserts/training_data/split_video_25fps",
                                 help='path of source video in 25 fps')
        self.parser.add_argument('--openface_landmark_dir', type=str, default="./asserts/training_data/split_video_25fps_landmark_openface",
                                 help='path of openface landmark dir')
        self.parser.add_argument('--video_frame_dir', type=str, default="./asserts/training_data/split_video_25fps_frame",
                                 help='path of video frames')
        self.parser.add_argument('--audio_dir', type=str, default="./asserts/training_data/split_video_25fps_audio",
                                 help='path of audios')
        self.parser.add_argument('--deep_speech_dir', type=str, default="./asserts/training_data/split_video_25fps_deepspeech",
                                 help='path of deep speech')
        self.parser.add_argument('--crop_face_dir', type=str, default="./asserts/training_data/split_video_25fps_crop_face",
                                 help='path of crop face dir')
        self.parser.add_argument('--json_path', type=str, default="./asserts/training_data/training_json.json",
                                 help='path of training json')
        self.parser.add_argument('--clip_length', type=int, default=9, help='clip length')
        self.parser.add_argument('--deep_speech_model', type=str, default="./asserts/output_graph.pb",
                                 help='path of pretrained deepspeech model')
        return self.parser.parse_args()

class LipSickInferenceOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="LipSick Inference Options")

        self.parser.add_argument("--source_channel", type=int, default=3, help="Number of channels in the source video")
        self.parser.add_argument("--ref_channel", type=int, default=15, help="Number of channels in the reference video")
        self.parser.add_argument("--audio_channel", type=int, default=29, help="Number of channels in the driving audio")
        self.parser.add_argument("--mouth_region_size", type=int, default=256, help="Size of the mouth region")
        self.parser.add_argument("--source_video_path", type=str, required=True, help="Path to the source video")
        self.parser.add_argument("--source_openface_landmark_path", type=str, default=None, help="Path to the OpenFace landmarks file")
        self.parser.add_argument("--driving_audio_path", type=str, required=True, help="Path to the driving audio file")
        self.parser.add_argument("--pretrained_lipsick_path", type=str, default="./asserts/pretrained_lipsick.pth", help="Path to the pretrained LipSick model")
        self.parser.add_argument("--deepspeech_model_path", type=str, default="./asserts/output_graph.pb", help="Path to the DeepSpeech model")
        self.parser.add_argument("--res_video_dir", type=str, default="./asserts/inference_result", help="Directory to save the resulting video")

        self.parser.add_argument("--custom_crop_radius", type=int, default=None, help="Custom crop radius for all frames")
        self.parser.add_argument("--custom_reference_frames", type=str, default=None, help="Comma-separated list of custom reference frame indices")
        self.parser.add_argument("--activate_custom_frames", action='store_true', help="Activate custom reference frames if set")

        self.parser.add_argument("--samelength_video_path", type=str, default="./asserts/inference_result/samelength.mp4", help="Path to the samelength video")
        self.parser.add_argument("--auto_mask", action='store_true', help="Generate a same-length video for auto masking")
        self.parser.add_argument("--pre_blend_video_path", type=str, default="./asserts/inference_result/pre_blend.mp4", help="Path to the pre-blend or lipsick video")
        
        # Set lipsick_video_path to the default value
        self.parser.add_argument('--lipsick_video_path', type=str, default="./asserts/inference_result/_lipsick.mp4", help="Path to the pre-blended or lipsick video")

    def parse_args(self):
        return self.parser.parse_args()
