#!/bin/bash

wget --no-check-certificate -P eval_ucf101/ https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
wget --no-check-certificate -P eval_ucf101/ https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

unrar x eval_ucf101/UCF101.rar eval_ucf101/
unzip eval_ucf101/UCF101TrainTestSplits-RecognitionTask.zip -d eval_ucf101/

rm eval_ucf101/UCF101.rar eval_ucf101/UCF101TrainTestSplits-RecognitionTask.zip

echo "Data successfully prepared in eval_ucf101/; deleted zip"