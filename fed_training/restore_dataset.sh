#!/bin/bash

# restore_from_hf.sh
set -e

REPO_ID="tiger9406/fed-client-dataset"
LOCAL_DIR="/content/vjepa2_federated/fed_training"
mkdir -p "$LOCAL_DIR"

# List files in the HF repo to identify what's available
echo "Fetching file list from $REPO_ID..."
FILES=$(python3 -c "
from huggingface_hub import HfApi
api = HfApi()
files = [f.rfilename for f in api.list_repo_files(repo_id='$REPO_ID', repo_type='dataset')]
print(' '.join([f for f in files if f.endswith('.tar.gz')]))
")

for FILE in $FILES; do
    CLIENT_NAME="${FILE%.tar.gz}"
    CLIENT_PATH="${LOCAL_DIR}/${CLIENT_NAME}"
    
    echo "--- Restoring $CLIENT_NAME ---"
    mkdir -p "$CLIENT_PATH"
    
    # Download archive
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$REPO_ID',
    filename='$FILE',
    repo_type='dataset',
    local_dir='$LOCAL_DIR',
    token=os.environ.get('HF_TOKEN')
)
"
    
    # Extract and clean up
    tar -xzf "${LOCAL_DIR}/${FILE}" -C "$CLIENT_PATH"
    rm "${LOCAL_DIR}/${FILE}"
    echo "Done. Extracted to $CLIENT_PATH"
done

echo "Restoration complete."