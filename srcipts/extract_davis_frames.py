"""将视频均匀采样为指定数量的 PNG 图片（默认 70 张）。

直接运行：python extract_davis_frames.py
依赖：Python 3，以及 PATH 中的 ffmpeg / ffprobe，无需安装 Python 第三方包。
源视频不足 70 帧时均匀重复邻近帧，不生成插值画面；对应关系见 frames_manifest.json。
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_VIDEO = Path(
    r"D:\A_Gorger\program\VIMT\carlsim6\CARLsim6-master\process\scripts"
    r"\davis_paraliging6_test4.avi"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "carlsimMT" / "davis_paraliging"


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Command failed: " + command[0])
    return result.stdout


def extract_frames(video, output, count=70):
    video = Path(video).resolve()
    output = Path(output).resolve()
    if count < 1:
        raise ValueError("Frame count must be positive.")
    if not video.is_file():
        raise FileNotFoundError(video)
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} was not found. Install FFmpeg and add its bin directory to PATH.")

    info = json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(video),
    ]))
    streams = info.get("streams", [])
    if not streams:
        raise ValueError("No video stream found.")
    stream = streams[0]
    total = int(stream["nb_read_frames"])
    if total < 1:
        raise ValueError("The video contains no decodable frames.")

    # 包含首尾帧；整数运算四舍五入，帧数不足时自然重复邻近帧。
    indices = [
        (2 * i * (total - 1) + count - 1) // (2 * (count - 1))
        if count > 1 else 0
        for i in range(count)
    ]
    unique_indices = sorted(set(indices))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("frame_*.png")) or (output / "frames_manifest.json").exists():
        raise FileExistsError(f"Output already contains extracted frames; choose an empty directory: {output}")

    # 先解码到临时目录，成功后再输出；重复采样直接复制 PNG，避免二次编码。
    with tempfile.TemporaryDirectory(prefix=".extract_", dir=output) as temporary:
        staging = Path(temporary)
        selection = "+".join(f"eq(n\\,{index})" for index in unique_indices)
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-map", "0:v:0", "-vf", "select=" + selection,
            "-fps_mode", "vfr", "-frames:v", str(len(unique_indices)),
            "-start_number", "1", str(staging / "source_%06d.png"),
        ])
        decoded = sorted(staging.glob("source_*.png"))
        if len(decoded) != len(unique_indices):
            raise RuntimeError(f"Expected {len(unique_indices)} decoded frames, got {len(decoded)}.")
        sources = dict(zip(unique_indices, decoded))
        frames = []
        for number, source_index in enumerate(indices, start=1):
            filename = f"frame_{number:03d}.png"
            shutil.copyfile(sources[source_index], output / filename)
            frames.append({"file": filename, "source_frame_1based": source_index + 1})

    manifest = {
        "source_video": str(video),
        "source_frame_count": total,
        "output_frame_count": count,
        "width": stream["width"],
        "height": stream["height"],
        "source_frame_rate": stream["r_frame_rate"],
        "sampling": "uniform nearest-frame sampling including first and last frames",
        "repeated_frame_count": count - len(unique_indices),
        "frames": frames,
    }
    (output / "frames_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Source: {total} frames, {stream['width']} x {stream['height']}")
    print(f"Saved {count} PNG images ({count - len(unique_indices)} repeated frames) to: {output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=70)
    args = parser.parse_args()
    try:
        extract_frames(args.video, args.output, args.count)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
