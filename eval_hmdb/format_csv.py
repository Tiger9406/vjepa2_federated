import os
import re

def sanitize(name):
    base, ext = os.path.splitext(name)
    base = re.sub(r'[&#+\(\)\[\];!\-@,\'\"?]', '', base)
    base = re.sub(r'_+', '_', base)   # collapse multiple underscores
    base = base.strip('_')
    return base + ext

def prepare_hmdb51(split_num=1):
    current_dir = os.getcwd()
    current_dir = os.path.join(current_dir, "eval_hmdb")
    video_dir = os.path.join(current_dir, "raw_hmdb", "hmdb51")
    split_dir = os.path.join(current_dir, "testTrainMulti_7030_splits")

    classes = sorted([
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    ])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    out_train = os.path.join(current_dir, f"hmdb51_train_split{split_num}.csv")
    out_val   = os.path.join(current_dir, f"hmdb51_val_split{split_num}.csv")

    total_train, total_val, total_warn = 0, 0, 0

    with open(out_train, "w") as f_train, open(out_val, "w") as f_val:
        for cls in classes:
            split_file = os.path.join(split_dir, f"{cls}_test_split{split_num}.txt")
            if not os.path.exists(split_file):
                print(f"WARNING: no split file for class '{cls}'")
                continue

            label = class_to_idx[cls]
            cls_dir = os.path.join(video_dir, cls)

            # Build sanitized lookup: sanitized_name -> original_name
            disk_files = {sanitize(f): f for f in os.listdir(cls_dir)
                          if os.path.isfile(os.path.join(cls_dir, f))}

            with open(split_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        print(f"WARNING: malformed line in {split_file}: {line!r}")
                        continue

                    fname, assignment = parts

                    # assignment "0" = not used in this split, skip
                    if assignment == "0":
                        continue

                    abs_path = os.path.join(cls_dir, fname)

                    # Try sanitized fallback if exact match not found
                    if not os.path.exists(abs_path):
                        sanitized_key = sanitize(fname)
                        matched = disk_files.get(sanitized_key)
                        if matched:
                            abs_path = os.path.join(cls_dir, matched)
                        else:
                            print(f"WARNING: no match for {fname} in class '{cls}'")
                            total_warn += 1
                            continue  # skip unresolvable files

                    if assignment == "1":
                        f_train.write(f"{abs_path} {label}\n")
                        total_train += 1
                    elif assignment == "2":
                        f_val.write(f"{abs_path} {label}\n")
                        total_val += 1
                    else:
                        print(f"WARNING: unknown assignment '{assignment}' for {fname}")

    print(f"Split {split_num}: {total_train} train, {total_val} val, {total_warn} unresolved")
    print(f"Generated: {out_train}")
    print(f"Generated: {out_val}")

if __name__ == "__main__":
    for split in [1, 2, 3]:
        prepare_hmdb51(split_num=split)