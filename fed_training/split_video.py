import csv
import math
import os
import subprocess


def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def split_video(video_path, output_dir, segment_duration=60):
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_duration(video_path)
    num_segments = math.ceil(duration / segment_duration)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_paths = []

    for i in range(num_segments):
        start = i * segment_duration
        out_path = os.path.join(output_dir, f"{base_name}_clip_{i:04d}.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-i",
                video_path,
                "-t",
                str(segment_duration),
                "-c",
                "copy",
                out_path,
            ],
            capture_output=True,
        )
        if os.path.exists(out_path):
            output_paths.append(out_path)
        print(f"  Segment {i + 1}/{num_segments}: {out_path}")

    return output_paths


def process_csv(input_csv, output_csv, output_dir, segment_duration=60):
    all_clips = []

    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows:
        if not row:
            continue
        video_path = row[0].strip().split()[0]
        if not os.path.exists(video_path):
            print(f"Skipping missing file: {video_path}")
            continue
        print(f"Splitting: {video_path}")
        clips = split_video(video_path, output_dir, segment_duration)
        all_clips.extend(clips)

    with open(output_csv, "w") as f:
        for clip in all_clips:
            f.write(f"{clip} 0\n")

    print(f"\nDone. {len(all_clips)} clips written to {output_csv}")


if __name__ == "__main__":
    process_csv(
        input_csv="/content/vjepa2_federated/fed_training/client_1/videos.csv",
        output_csv="/content/vjepa2_federated/fed_training/client_1/videos.csv",
        output_dir="/content/vjepa2_federated/fed_training/client_1/clips/",
        segment_duration=20,
    )
