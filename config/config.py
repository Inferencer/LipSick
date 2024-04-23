import argparse

class DataProcessingOptions():
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



class LipSickInferenceOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument('--source_channel', type=int, default=3, help='channels of source image')
        self.parser.add_argument('--ref_channel', type=int, default=15, help='channels of reference image')
        self.parser.add_argument('--audio_channel', type=int, default=29, help='channels of audio feature')
        self.parser.add_argument('--mouth_region_size', type=int, default=256, help='help to resize window')
        self.parser.add_argument('--source_video_path',
                                 default='./asserts/examples/test4.mp4',
                                 type=str,
                                 help='path of source video')
        self.parser.add_argument('--source_openface_landmark_path',
                                 default='./asserts/examples/test4.csv',
                                 type=str,
                                 help='path of detected openface landmark')
        self.parser.add_argument('--driving_audio_path',
                                 default='./asserts/examples/driving_audio_1.wav',
                                 type=str,
                                 help='path of driving audio')
        self.parser.add_argument('--pretrained_lipsick_path',
                                 default='./asserts/pretrained_lipsick.pth',
                                 type=str,
                                 help='pretrained model of lipsick')
        self.parser.add_argument('--deepspeech_model_path',
                                 default='./asserts/output_graph.pb',
                                 type=str,
                                 help='path of deepspeech model')
        self.parser.add_argument('--res_video_dir',
                                 default='./asserts/inference_result',
                                 type=str,
                                 help='path of generated videos')
        return self.parser.parse_args()
