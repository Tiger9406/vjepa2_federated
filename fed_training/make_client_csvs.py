import os
import yaml
from pathlib import Path

def generate_video_csv(source_dir, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    valid_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    video_paths = []

    for root, _, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix.lower() in valid_extensions:
                full_path = os.path.abspath(os.path.join(root, file))
                video_paths.append(full_path)

    if not video_paths:
        print(f"Warning: No video files found in {source_dir}")
        return

    # space separated
    with open(output_csv, mode='w') as f:
        for path in video_paths:
            f.write(f"{path} 0\n")

    print(f"Generated {output_csv} with {len(video_paths)} videos.")

if __name__ == "__main__":
    yaml_path = "./fed_training/fed_vits.yaml"
    print(f"Loading config from {yaml_path}...")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    output_csvs = config['data']['client_datasets']
    source_dirs = config['data']['client_datasets_path']

    for i, (source, output) in enumerate(zip(source_dirs, output_csvs)):
        print(f"\nProcessing Client {i}")
        generate_video_csv(source, output)