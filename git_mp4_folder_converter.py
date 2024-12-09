'''
python git_mp4_folder_converter.py input_folder output_folder --mode gif2mp4

python git_mp4_folder_converter.py input_folder output_folder --mode mp42gif
'''
#!/usr/bin/env python
# coding: utf-8

import argparse
from moviepy.editor import VideoFileClip, ImageSequenceClip
import os

def gif_to_mp4(input_gif, output_mp4):
    """
    将 GIF 文件转换为 MP4 文件。

    :param input_gif: 输入 GIF 文件路径
    :param output_mp4: 输出 MP4 文件路径
    """
    clip = VideoFileClip(input_gif)
    clip.write_videofile(output_mp4, codec='libx264')
    print(f"GIF 文件已转换为 MP4 文件并保存到 {output_mp4}")

def mp4_to_gif(input_mp4, output_gif):
    """
    将 MP4 文件转换为 GIF 文件。

    :param input_mp4: 输入 MP4 文件路径
    :param output_gif: 输出 GIF 文件路径
    """
    clip = VideoFileClip(input_mp4)
    clip.write_gif(output_gif)
    print(f"MP4 文件已转换为 GIF 文件并保存到 {output_gif}")

def process_folder(input_folder, output_folder, mode):
    """
    处理文件夹中的所有 GIF 或 MP4 文件，并将其转换为相应的格式。

    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param mode: 转换模式 (gif2mp4 或 mp42gif)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        if mode == "gif2mp4" and filename.lower().endswith('.gif'):
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.mp4')
            gif_to_mp4(input_file, output_file)
        elif mode == "mp42gif" and filename.lower().endswith('.mp4'):
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.gif')
            mp4_to_gif(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 GIF 文件与 MP4 文件进行互转")
    parser.add_argument("input_folder", help="输入文件夹路径")
    parser.add_argument("output_folder", help="输出文件夹路径")
    parser.add_argument("--mode", choices=["gif2mp4", "mp42gif"], required=True, help="转换模式 (gif2mp4 或 mp42gif)")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.mode)
