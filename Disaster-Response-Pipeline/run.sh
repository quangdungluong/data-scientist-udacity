#!/bin/bash
echo "Run process data and train model"
python data/process_data.py
python models/train_classifier.py
echo "Successfully"