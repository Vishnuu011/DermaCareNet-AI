import os
import sys

from src.DermaCareNet.exception import ComputerVisionYolov5Exception

from typing import Optional, List, Literal

import gdown

import zipfile

import yaml

from pathlib import Path
import textwrap
import subprocess



class Yolov5TrainingPipeline:

    """
    Pipeline for automating YOLOv5 dataset download, 
    config generation, and training.
    """

    def __init__(self, url: str, prefix: str) -> None:

        """
        Initialize the YOLOv5 training pipeline.

        Args:
            url (str): Google Drive or dataset URL.
            prefix (str): URL prefix for Google Drive file download.
        """

        self.url = url
        self.prefix = prefix

    def download_data_from_drive(self, url: str,  prefix: str) -> None:

        """
        Download dataset from Google Drive and extract it.

        Args:
            url (str): Google Drive URL of the dataset.
            prefix (str): Prefix for constructing the gdown download link.

        Raises:
            ComputerVisionYolov5Exception: If download or extraction fails.
        """

        try:
            file_id = url.split("/")[-2]
            print(f"url file id: {file_id}")

            output_zip = "yolov5pytorch.zip"

            gdown.download(
                prefix + file_id, 
                output_zip
            )

            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(".")

            os.remove(output_zip)    

        except Exception as e:
            raise ComputerVisionYolov5Exception(e, sys)
        
    def extract_num_class_in_yaml(self) -> int:

        """
        Extract the number of classes (`nc`) from `data.yaml`.

        Returns:
            int: Number of classes defined in data.yaml.

        Raises:
            ComputerVisionYolov5Exception: If YAML parsing fails.
        """

        try:
            with open("data.yaml", "r") as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            return num_classes   
         
        except Exception as e:
            raise ComputerVisionYolov5Exception(e, sys)    
        

    def generate_yolov5m_config(self, num_classes: int, filename: str = "custom_yolov5m.yaml") -> None:

        """
        Generate a custom YOLOv5m model config with the specified number of classes.

        Args:
            num_classes (int): Number of classes for detection.
            filename (str, optional): Output config filename. Defaults to "custom_yolov5m.yaml".

        Raises:
            ComputerVisionYolov5Exception: If config generation fails.
        """

        try:
            config_content = textwrap.dedent(f"""\
                # Ultralytics AGPL-3.0 License - https://ultralytics.com/license

                # Parameters
                nc: {num_classes} # number of classes
                depth_multiple: 0.67 # model depth multiple
                width_multiple: 0.75 # layer channel multiple
                anchors:
                  - [10, 13, 16, 30, 33, 23] # P3/8
                  - [30, 61, 62, 45, 59, 119] # P4/16
                  - [116, 90, 156, 198, 373, 326] # P5/32                

                # YOLOv5 v6.0 backbone
                backbone:
                  # [from, number, module, args]
                  [
                    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
                    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
                    [-1, 3, C3, [128]],
                    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
                    [-1, 6, C3, [256]],
                    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
                    [-1, 9, C3, [512]],
                    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
                    [-1, 3, C3, [1024]],
                    [-1, 1, SPPF, [1024, 5]], # 9
                  ]                

                # YOLOv5 v6.0 head
                head: [
                    [-1, 1, Conv, [512, 1, 1]],
                    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
                    [[-1, 6], 1, Concat, [1]], # cat backbone P4
                    [-1, 3, C3, [512, False]], # 13                

                    [-1, 1, Conv, [256, 1, 1]],
                    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
                    [[-1, 4], 1, Concat, [1]], # cat backbone P3
                    [-1, 3, C3, [256, False]], # 17 (P3/8-small)                

                    [-1, 1, Conv, [256, 3, 2]],
                    [[-1, 14], 1, Concat, [1]], # cat head P4
                    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)                

                    [-1, 1, Conv, [512, 3, 2]],
                    [[-1, 10], 1, Concat, [1]], # cat head P5
                    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)                

                    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
                  ]
                """)

            output_path = Path("yolov5") / "models" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(config_content)

            print(f"✅ Config saved to: {output_path}")
        except Exception as e:
            raise ComputerVisionYolov5Exception(e, sys)
        
    def yolo5m_training(self, img_size: int, batch: int, epochs: int, 
                    data_yaml: str, yolo5m: str) -> None:
        
        """
        Train YOLOv5m model with custom dataset.

        Args:
            img_size (int): Training image size.
            batch (int): Batch size for training.
            epochs (int): Number of training epochs.
            data_yaml (str): Path to data.yaml file.
            yolo5m (str): Pretrained YOLOv5m weights file.

        Raises:
            ComputerVisionYolov5Exception: If training fails.
        """

        try:
            cmd = textwrap.dedent(f"""
                cd yolov5 && \
                python train.py --img {img_size} \
                --batch {batch} \
                --epochs {epochs} \
                --data {data_yaml} \
                --cfg ./models/custom_yolov5m.yaml \
                --weights {yolo5m} \
                --name yolov5m_640_results \
                --cache
            """)

            print(f"Running YOLOv5 training:\n{cmd}")
            subprocess.run(cmd, shell=True, check=True)

        except subprocess.CalledProcessError as e:
            print("Training failed:", e)
            raise ComputerVisionYolov5Exception(e, sys)
        except Exception as e:
            raise ComputerVisionYolov5Exception(e, sys)  
         

    def initialize_pipeline(self) -> None:

        """
        Initialize the YOLOv5 training pipeline.

        Workflow:
            1. Download dataset from Google Drive.
            2. Extract number of classes from data.yaml.
            3. Generate YOLOv5m config.
            4. Start training.

        Raises:
            ComputerVisionYolov5Exception: If any step in pipeline fails.
        """

        try:
            self.download_data_from_drive(self.url, self.prefix)
            num_classes=self.extract_num_class_in_yaml()
            self.generate_yolov5m_config(num_classes=num_classes)
            self.yolo5m_training(
                img_size=640,
                batch=8,
                epochs=2,
                data_yaml='../data.yaml',
                yolo5m='yolov5m.pt'
            )
        except Exception as e:
            raise ComputerVisionYolov5Exception(e, sys)    




