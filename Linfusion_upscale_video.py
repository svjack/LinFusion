#!/usr/bin/env python
# coding: utf-8
import argparse
from moviepy.editor import VideoFileClip, ImageSequenceClip
from src.pipelines.pipeline_superres_sdxl import StableDiffusionXLSuperResPipeline
from diffusers import AutoPipelineForText2Image
import torch
import os

from PIL import Image
from tqdm import tqdm

from src.tools import (
    forward_unet_wrapper, 
    forward_resnet_wrapper, 
    forward_crossattndownblock2d_wrapper, 
    forward_crossattnupblock2d_wrapper,
    forward_downblock2d_wrapper, 
    forward_upblock2d_wrapper,
    forward_transformer_block_wrapper
)
from src.linfusion import LinFusion

def initialize_pipe(device):
    model_ckpt = "svjack/GenshinImpact_XL_Base"
    pipe = StableDiffusionXLSuperResPipeline.from_pretrained(
        model_ckpt, torch_dtype=torch.float16
    ).to(device)

    linfusion = LinFusion.construct_for(pipe, pretrained_model_name_or_path="Yuanshi/LinFusion-XL")
    pipe.enable_vae_tiling()

    return pipe

def upscale_image(pipe, image, prompt, width, height, device, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, gaussian_sigma, upscale_strength):
    generator = torch.manual_seed(0)
    upscaled_image = pipe(
        image=image, prompt=prompt,
        height=height, width=width, device=device, 
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3, gaussian_sigma=gaussian_sigma,
        generator=generator, upscale_strength=upscale_strength
    ).images[0]

    return upscaled_image

def main(mp4_path, prompt, width, height, upscale_factor, output_filename, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, gaussian_sigma, upscale_strength):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if upscale_factor:
        video_clip = VideoFileClip(mp4_path)
        width = int(video_clip.w * upscale_factor)
        height = int(video_clip.h * upscale_factor)

    # Initialize the pipeline
    pipe = initialize_pipe(device)

    # Load the video
    video_clip = VideoFileClip(mp4_path)
    frames = [frame for frame in video_clip.iter_frames()]

    # Upscale each frame
    upscaled_frames = []
    for frame in tqdm(frames, desc="Upscaling frames"):
        image = Image.fromarray(frame)
        upscaled_image = upscale_image(pipe, image, prompt, width, height, device, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, gaussian_sigma, upscale_strength)
        upscaled_frames.append(upscaled_image)

    # Save the upscaled frames to a temporary directory
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    for i, frame in enumerate(upscaled_frames):
        frame.save(os.path.join(temp_dir, f"frame_{i:04d}.png"))

    # Create a new video from the upscaled frames
    if not output_filename:
        input_filename = os.path.basename(mp4_path)
        input_name, _ = os.path.splitext(input_filename)
        output_filename = f"{input_name}_upscaled_{width}x{height}.mp4"

    upscaled_video_path = output_filename
    frame_paths = [os.path.join(temp_dir, f"frame_{i:04d}.png") for i in range(len(upscaled_frames))]
    upscaled_clip = ImageSequenceClip(frame_paths, fps=video_clip.fps)
    upscaled_clip.write_videofile(upscaled_video_path, codec='libx264')

    # Clean up temporary directory
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    print(f"Upscaled video saved to {upscaled_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale video frames using Stable Diffusion XL.")
    parser.add_argument("mp4_path", type=str, help="Path to the input MP4 file.")
    parser.add_argument("prompt", type=str, help="Prompt for the image generation.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the upscaled image.")
    parser.add_argument("--height", type=int, default=512, help="Height of the upscaled image.")
    parser.add_argument("--upscale_factor", type=int, default=None, help="Upscale factor to automatically calculate width and height.")
    parser.add_argument("--output_filename", type=str, default=None, help="Output filename for the upscaled video.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--cosine_scale_1", type=float, default=3, help="Cosine scale 1.")
    parser.add_argument("--cosine_scale_2", type=float, default=1, help="Cosine scale 2.")
    parser.add_argument("--cosine_scale_3", type=float, default=1, help="Cosine scale 3.")
    parser.add_argument("--gaussian_sigma", type=float, default=0.8, help="Gaussian sigma.")
    parser.add_argument("--upscale_strength", type=float, default=0.32, help="Upscale strength.")

    args = parser.parse_args()
    main(args.mp4_path, args.prompt, args.width, args.height, args.upscale_factor, args.output_filename, args.num_inference_steps, args.guidance_scale, args.cosine_scale_1, args.cosine_scale_2, args.cosine_scale_3, args.gaussian_sigma, args.upscale_strength)
