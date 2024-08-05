import os,sys,io
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir,"anitalker"))

import time
import folder_paths
import cuda_malloc
output_dir = folder_paths.get_output_directory()
pretrained_dir = os.path.join(now_dir,"pretrained_models")
device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"

import cv2
import numpy as np
from PIL import Image
import torch
import torchaudio
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet

from huggingface_hub import snapshot_download

from anitalker.templates import ffhq256_autoenc,LitModel,th
from anitalker.LIA_Model import LIA_Model

import python_speech_features
from torchvision import transforms
from moviepy.editor import *

import shutil
from tqdm import tqdm

def img_preprocessing(img,img_size):
    # detect face
    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet18", pretrained=True, num_nb=10, num_lms=68, 
            net_stride=32, input_size=256, meanface_type="300w", 
            map_location=device, checkpoint=None
        )
    )
    frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    landmarks, bboxes = torchlm.runtime.forward(frame)
                    
    if bboxes.shape == (1, 5):
        bbox = bboxes[0]
    elif bboxes.shape[0] > 0:
        # If multiple persons exist, select the one with the largest width
        bbox = max(bboxes, key=lambda bbox: bbox[2] - bbox[0])
    
    x1, y1, x2, y2 = bbox[:4]
    src_width = img.size[1]
    src_height = img.size[0]

    width = x2 - x1
    height = y2 - y1

    pad = abs(width - height)
    if width > height:
        y1 = max(0, y1 - pad / 2)
        y2 = min(src_height, y2 + pad / 2)
    else:
        x1 = max(0, x1 - pad / 2)
        x2 = min(src_width, x2 + pad / 2)
    
    face_img = img.crop((x1,y1,x2,y2))
    # face_img.save("test.jpg")
    face_img = face_img.resize((img_size,img_size))
    face_img = np.asarray(face_img)
    face_img = np.transpose(face_img, (2, 0, 1))  # 3 x 256 x 256
    face_img = face_img / 255.0
    img_tensor = torch.from_numpy(face_img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img_tensor - 0.5) * 2.0 
    return imgs_norm

def saved_image(img_tensor, img_path):
    toPIL = transforms.ToPILImage()
    img = toPIL(img_tensor.detach().cpu().squeeze(0))  # 使用squeeze(0)来移除批次维度
    img.save(img_path)

def frames_to_video(input_path, audio_path, output_path, fps=25):
    image_files = [os.path.join(input_path, img) for img in sorted(os.listdir(input_path))]
    clips = [ImageClip(m).set_duration(1/fps) for m in image_files]
    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')


class PreViewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "AIFSH_AniTalker"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}


prompt_sr = 16000
class AniTalkerNode:
    def __init__(self) -> None:
        if os.path.isfile(os.path.join(pretrained_dir,"stage1.ckpt")):
            print("use cache models,make sure your pretrained_models complete")
        else:
            snapshot_download(repo_id="taocode/anitalker_ckpts",local_dir=pretrained_dir)
        
        self.infer_type = None
        self.lia = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":("IMAGE",),
                "audio":("AUDIO",),
                "infer_type":(["mfcc_full_control","mfcc_pose_only",
                               "hubert_pose_only","hubert_audio_only",
                               "hubert_full_control"],),
                "pose_yaw":("FLOAT",{
                    "default":0.25,
                }),
                "pose_pitch":("FLOAT",{
                    "default":0,
                }),
                "pose_roll":("FLOAT",{
                    "default":0,
                }),
                "face_location":("FLOAT",{
                    "default":0.5,
                }),
                "face_scale":("FLOAT",{
                    "default":0.5,
                }),
                "face_sr":("BOOLEAN",{
                    "default": False
                }),
                "control_flag":("BOOLEAN",{
                    "default": False
                }),
                "step_T":("INT",{
                    "default": 42
                }),
                "seed":("INT",{
                    "default": 42
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_AniTalker"

    def comfyimage2Image(self,comfyimage):
        comfyimage = comfyimage.numpy()[0] * 255
        image_np = comfyimage.astype(np.uint8)
        image = Image.fromarray(image_np)
        return image

    def generate(self,image,audio,infer_type,pose_yaw,pose_pitch,
                 pose_roll,face_location,face_scale,face_sr,control_flag,step_T,seed):
        frames_result_saved_path = os.path.join(output_dir, "anitalker",'frames')
        os.makedirs(frames_result_saved_path, exist_ok=True)
        
        timestimp = time.time_ns()
        predicted_video_256_path = os.path.join(output_dir,  f'anitalker-{timestimp}.mp4')
        predicted_video_512_path = os.path.join(output_dir,  f'anitalker_SR-{timestimp}.mp4')

        #======Loading Stage 1 model=========
        if self.lia is None:
            self.lia = LIA_Model(motion_dim=20, fusion_type='weighted_sum')
            self.lia.load_lightning_model(os.path.join(pretrained_dir,"stage1.ckpt"))
            self.lia.to(device)
        conf = ffhq256_autoenc()
        conf.seed = seed
        conf.decoder_layers = 2
        conf.infer_type = infer_type
        conf.motion_dim = 20

        if infer_type == 'mfcc_full_control':
            conf.face_location=True
            conf.face_scale=True
            conf.mfcc = True

        elif infer_type == 'mfcc_pose_only':
            conf.face_location=False
            conf.face_scale=False
            conf.mfcc = True

        elif infer_type == 'hubert_pose_only':
            conf.face_location=False
            conf.face_scale=False
            conf.mfcc = False

        elif infer_type == 'hubert_audio_only':
            conf.face_location=False
            conf.face_scale=False
            conf.mfcc = False

        elif infer_type == 'hubert_full_control':
            conf.face_location=True
            conf.face_scale=True
            conf.mfcc = False

        else:
            print('Type NOT Found!')
            exit(0)
        
        img_source = img_preprocessing(self.comfyimage2Image(image), 256).to(device)
        one_shot_lia_start, one_shot_lia_direction, feats = self.lia.get_start_direction_code(img_source, img_source, img_source, img_source)
        
        #======Loading Stage 2 model=========
        if self.infer_type != infer_type:
            name_list = infer_type.split("_")
            model_name = "_".join(["stage2"] + name_list[1:] + [name_list[0] + ".ckpt"])
            model_path = os.path.join(pretrained_dir, model_name)
            self.model = LitModel(conf)
            state = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state, strict=True)
            self.model.ema_model.eval()
            self.model.ema_model.to(device)
            self.infer_type = infer_type
            print(f"Load stage2 model from {model_path}")
        #=================================
        #======Audio Input=========
        waveform = audio['waveform'].squeeze(0)
        source_sr = audio['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        if conf.infer_type.startswith('mfcc'):
            # MFCC features
            input_values = python_speech_features.mfcc(signal=speech, samplerate=prompt_sr, numcep=13, winlen=0.025, winstep=0.01)
            d_mfcc_feat = python_speech_features.base.delta(input_values, 1)
            d_mfcc_feat2 = python_speech_features.base.delta(input_values, 2)
            audio_driven_obj = np.hstack((input_values, d_mfcc_feat, d_mfcc_feat2))
            frame_start, frame_end = 0, int(audio_driven_obj.shape[0]/4)
            audio_start, audio_end = int(frame_start * 4), int(frame_end * 4) # The video frame is fixed to 25 hz and the audio is fixed to 100 hz
            
            audio_driven = torch.Tensor(audio_driven_obj[audio_start:audio_end,:]).unsqueeze(0).float().to(device)
            
        elif conf.infer_type.startswith('hubert'):
            hubert_model_path = os.path.join(pretrained_dir, 'chinese-hubert-large')
            if not os.path.exists(hubert_model_path):
                print('Please download the hubert weight into the ckpts path first.')
                exit(0)
            print('You did not extract the audio features in advance, extracting online now, which will increase processing delay')

            start_time = time.time()

            # load hubert model
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            audio_model = HubertModel.from_pretrained(hubert_model_path).to(device)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_path)
            audio_model.feature_extractor._freeze_parameters()
            audio_model.eval()

            # hubert model forward pass
            # print(speech.shape)
            input_values = feature_extractor(speech.squeeze(0), sampling_rate=16000, padding=True, do_normalize=True, return_tensors="pt").input_values
            input_values = input_values.to(device)
            ws_feats = []
            with torch.no_grad():
                outputs = audio_model(input_values, output_hidden_states=True)
                for i in range(len(outputs.hidden_states)):
                    ws_feats.append(outputs.hidden_states[i].detach().cpu().numpy())
                ws_feat_obj = np.array(ws_feats)
                ws_feat_obj = np.squeeze(ws_feat_obj, 1)
                ws_feat_obj = np.pad(ws_feat_obj, ((0, 0), (0, 1), (0, 0)), 'edge') # align the audio length with video frame
            
            execution_time = time.time() - start_time
            print(f"Extraction Audio Feature: {execution_time:.2f} Seconds")

            audio_driven_obj = ws_feat_obj
            

            frame_start, frame_end = 0, int(audio_driven_obj.shape[1]/2)
            audio_start, audio_end = int(frame_start * 2), int(frame_end * 2) # The video frame is fixed to 25 hz and the audio is fixed to 50 hz
            
            audio_driven = torch.Tensor(audio_driven_obj[:,audio_start:audio_end,:]).unsqueeze(0).float().to(device)
        #============================
        
        # Diffusion Noise
        noisyT = th.randn((1,frame_end, 20)).to(device)
        
        #======Inputs for Attribute Control=========
        yaw_signal = torch.zeros(1, frame_end, 1).to(device) + pose_yaw
        pitch_signal = torch.zeros(1, frame_end, 1).to(device) + pose_pitch
        roll_signal = torch.zeros(1, frame_end, 1).to(device) + pose_roll
        pose_signal = torch.cat((yaw_signal, pitch_signal, roll_signal), dim=-1)

        pose_signal = torch.clamp(pose_signal, -1, 1)

        face_location_signal = torch.zeros(1, frame_end, 1).to(device) + face_location
        face_scae_signal = torch.zeros(1, frame_end, 1).to(device) + face_scale
        #===========================================

        start_time = time.time()

        #======Diffusion Denosing Process=========
        generated_directions = self.model.render(one_shot_lia_start, one_shot_lia_direction, audio_driven, face_location_signal, face_scae_signal, pose_signal, noisyT, step_T, control_flag=control_flag)
        #=========================================
        
        execution_time = time.time() - start_time
        print(f"Motion Diffusion Model: {execution_time:.2f} Seconds")

        generated_directions = generated_directions.detach().cpu().numpy()
        
        start_time = time.time()
        #======Rendering images frame-by-frame=========
        for pred_index in tqdm(range(generated_directions.shape[1])):
            ori_img_recon = self.lia.render(one_shot_lia_start, torch.Tensor(generated_directions[:,pred_index,:]).to(device), feats)
            ori_img_recon = ori_img_recon.clamp(-1, 1)
            wav_pred = (ori_img_recon.detach() + 1) / 2
            saved_image(wav_pred, os.path.join(frames_result_saved_path, "%06d.png"%(pred_index)))
        #==============================================
        
        execution_time = time.time() - start_time
        print(f"Renderer Model: {execution_time:.2f} Seconds")

        audio_path = os.path.join(output_dir,"tmp.flac")
        torchaudio.save(audio_path,speech,prompt_sr,format="FLAC")
        frames_to_video(frames_result_saved_path, audio_path, predicted_video_256_path)
        
        shutil.rmtree(frames_result_saved_path)
        
        res_video = predicted_video_256_path
        # Enhancer
        # Code is modified from https://github.com/OpenTalker/SadTalker/blob/cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8/src/utils/face_enhancer.py#L26

        if face_sr:
            from anitalker.face_sr.face_enhancer import enhancer_list
            import imageio

            # Super-resolution
            imageio.mimsave(predicted_video_512_path+'.tmp.mp4', enhancer_list(predicted_video_256_path, method='gfpgan', bg_upsampler=None), fps=float(25))
            
            # Merge audio and video
            video_clip = VideoFileClip(predicted_video_512_path+'.tmp.mp4')
            audio_clip = AudioFileClip(predicted_video_256_path)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(predicted_video_512_path, codec='libx264', audio_codec='aac')
            
            os.remove(predicted_video_512_path+'.tmp.mp4')
            res_video = predicted_video_512_path

        return (res_video,)
    

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "PreViewVideo":PreViewVideo,
    "AniTalkerNode": AniTalkerNode
}


 
     