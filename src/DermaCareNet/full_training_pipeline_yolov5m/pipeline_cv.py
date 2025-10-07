import os
import sys


from src.DermaCareNet.exception import ComputerVisionYolov11Exception

from ultralytics import YOLO

import torch


from typing import Optional, List, Literal

import gdown

import zipfile

import yaml

from pathlib import Path
import textwrap
import subprocess






class Yolov11TrainingPipeline:

    """
    Pipeline for automating YOLOv11 dataset download, 
    config generation, and training.
    """

    def __init__(self, url: str, prefix: str) -> None:

        """
        Initialize the YOLOv11 training pipeline.

        Args:
            url - str: Google Drive or dataset URL.
            prefix - str: URL prefix for Google Drive file download.
        """

        self.url = url
        self.prefix = prefix

    def download_data_from_drive(self, url: str,  prefix: str) -> None:

        """
        Download dataset from Google Drive and extract it.

        Args:
            url - str: Google Drive URL of the dataset.
            prefix - str: Prefix for constructing the gdown download link.

        Raises:
            ComputerVisionYolov11Exception: If download or extraction fails.
        """

        try:
            file_id = url.split("/")[-2]
            print(f"url file id: {file_id}")

            output_zip = "yolov11pytorch.zip"

            gdown.download(
                prefix + file_id, 
                output_zip
            )

            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(".")

            os.remove(output_zip)    

        except Exception as e:
            raise ComputerVisionYolov11Exception(e, sys)
        
    def extract_num_class_in_yaml(self) -> int:

        """
        Extract the number of classes (`nc`) from `data.yaml`.

        Returns:
            int: Number of classes defined in data.yaml.

        Raises:
            ComputerVisionYolov11Exception: If YAML parsing fails.
        """

        try:
            with open("data.yaml", "r") as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            return num_classes   
         
        except Exception as e:
            raise ComputerVisionYolov11Exception(e, sys)    
        

    def yolo11x_training(self, img_size: int, batch: int, epochs: int, data_yaml: str) -> None:

        """
        Train YOLOv11 model with custom dataset.
        """

        try:
            model = YOLO("yolo11x.pt") 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🚀 Using device: {device}")

            print("Running YOLOv11x training...")
            results = model.train(
                data=data_yaml,
                imgsz=img_size,
                batch=batch,
                epochs=epochs,
                name="yolov11x_640_results",
                device=device,
                workers=2,      
                cache=True
            )

            print("Training completed successfully!")

        except Exception as e:
            print("Training failed:", e)
            raise ComputerVisionYolov11Exception(e, sys.exc_info())

    def initialize_pipeline(self) -> None:

        """
        Initialize the YOLOv11x training pipeline.
        """

        try:
            self.download_data_from_drive(self.url, self.prefix)
            num_classes = self.extract_num_class_in_yaml()
            print(f"Number of classes: {num_classes}")

            self.yolo11x_training(
                img_size=640,
                batch=8,
                epochs=100,
                data_yaml="data.yaml"
            )

        except Exception as e:
            raise ComputerVisionYolov11Exception(e, sys.exc_info())  




