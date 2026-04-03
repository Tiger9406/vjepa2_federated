import os

def prepare_ucf101():
    current_dir = os.getcwd()
    
    video_dir = os.path.join(current_dir, "eval_ucf101", "UCF-101")
    split_dir = os.path.join(current_dir, "eval_ucf101", "ucfTrainTestlist")
    
    train_list = os.path.join(split_dir, "trainlist01.txt")
    test_list = os.path.join(split_dir, "testlist01.txt")
    
    out_train = os.path.join(current_dir, "eval_ucf101", "ucf101_train.csv")
    out_val = os.path.join(current_dir, "eval_ucf101", "ucf101_val.csv")
    
    with open(train_list, "r") as f_in, open(out_train, "w") as f_out:
        for line in f_in:
            path, label = line.strip().split(" ")
            abs_path = os.path.join(video_dir, path)
            f_out.write(f"{abs_path} {int(label) - 1}\n")
            
    class_mapping = {}
    with open(os.path.join(split_dir, "classInd.txt"), "r") as f:
        for line in f:
            label, class_name = line.strip().split(" ")
            class_mapping[class_name] = int(label) - 1
            
    with open(test_list, "r") as f_in, open(out_val, "w") as f_out:
        for line in f_in:
            path = line.strip()
            class_name = path.split("/")[0]
            label = class_mapping[class_name]
            abs_path = os.path.join(video_dir, path)
            f_out.write(f"{abs_path} {label}\n")

    print(f"Generated CSVs with absolute paths rooted at: {current_dir}")

if __name__ == "__main__":
    prepare_ucf101()