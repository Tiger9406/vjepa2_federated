import os
import re

def sanitize(name):
    # Remove special chars that ffmpeg/unrar strips
    return re.sub(r'[&#+\(\)\[\];!\-]', '', name).replace('__', '_').replace('--', '-')
def prepare_hmdb51(split_num=1):
    current_dir = os.getcwd()
    video_dir = os.path.join(current_dir, "raw_hmdb", "hmdb51")
    split_dir = os.path.join(current_dir, "testTrainMulti_7030_splits")

    classes = sorted([
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    ])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    out_train = os.path.join(current_dir, f"hmdb51_train_split{split_num}.csv")
    out_val   = os.path.join(current_dir, f"hmdb51_val_split{split_num}.csv")

    with open(out_train, "w") as f_train, open(out_val, "w") as f_val:
        for cls in classes:
            split_file = os.path.join(split_dir, f"{cls}_test_split{split_num}.txt")
            if not os.path.exists(split_file):
                continue
            label = class_to_idx[cls]

            cls_dir = os.path.join(video_dir, cls)
            disk_files = {sanitize(f): f for f in os.listdir(cls_dir)}
            with open(split_file) as f:
                for line in f:
                    fname, assignment = line.strip().split()
                    abs_path = os.path.join(cls_dir, fname)
                    if not os.path.exists(abs_path):
                      sanitized_key = sanitize(fname)
                      matched = disk_files.get(sanitized_key)
                      if matched:
                          abs_path = os.path.join(cls_dir, matched)
                          if assignment == "1":
                              f_train.write(f"{abs_path} {label}\n")
                          elif assignment == "2":
                              f_val.write(f"{abs_path} {label}\n")
                      else:
                          print(f"WARNING: no match for {fname}")

    print(f"Generated train/val CSVs for split {split_num}")

if __name__ == "__main__":
    prepare_hmdb51(split_num=1)