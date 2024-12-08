'''
### 示例 1：将 GIF 转换为 MP4（自动推断输出文件名）
```bash
python gif_mp4_converter.py input.gif --mode gif2mp4
```

### 示例 2：将 GIF 转换为 MP4（手动指定输出文件名）
```bash
python gif_mp4_converter.py input.gif --mode gif2mp4 --output_file output.mp4
```

### 示例 3：将 MP4 转换为 GIF（自动推断输出文件名）
```bash
python gif_mp4_converter.py input.mp4 --mode mp42gif
```

### 示例 4：将 MP4 转换为 GIF（手动指定输出文件名）
```bash
python gif_mp4_converter.py input.mp4 --mode mp42gif --output_file output.gif
```
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

def infer_output_file(input_file, mode):
    """
    根据输入文件和模式推断输出文件路径。

    :param input_file: 输入文件路径
    :param mode: 转换模式 (gif2mp4 或 mp42gif)
    :return: 推断的输出文件路径
    """
    base_name, ext = os.path.splitext(input_file)
    if mode == "gif2mp4":
        return f"{base_name}.mp4"
    elif mode == "mp42gif":
        return f"{base_name}.gif"
    else:
        raise ValueError("无效的转换模式")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 GIF 文件与 MP4 文件进行互转")
    parser.add_argument("input_file", help="输入文件路径 (GIF 或 MP4)")
    parser.add_argument("--output_file", help="输出文件路径 (GIF 或 MP4)")
    parser.add_argument("--mode", choices=["gif2mp4", "mp42gif"], required=True, help="转换模式 (gif2mp4 或 mp42gif)")

    args = parser.parse_args()

    # 如果未指定 output_file，则根据 input_file 和 mode 推断 output_file
    if not args.output_file:
        args.output_file = infer_output_file(args.input_file, args.mode)

    if args.mode == "gif2mp4":
        gif_to_mp4(args.input_file, args.output_file)
    elif args.mode == "mp42gif":
        mp4_to_gif(args.input_file, args.output_file)
