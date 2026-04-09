#!/bin/bash

set -e

REPO_ID="tiger9406/fed-client-dataset"
LOCAL_DIR="/content/vjepa2_federated/eval_hmdb"

python3 -c "
import os
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$REPO_ID',
    filename='latest.pt',
    repo_type='dataset',
    local_dir='$LOCAL_DIR',
    token=os.environ.get('HF_TOKEN')
)
"

echo "Done. Downloaded to $LOCAL_DIR/latest.pt"
python3 -c "
import os
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$REPO_ID',
    filename='fed_latest.pt',
    repo_type='dataset',
    local_dir='$LOCAL_DIR',
    token=os.environ.get('HF_TOKEN')
)
"
echo "Done. Downloaded to $LOCAL_DIR/fed_latest.pt"