import os
import sys

from src.DermaCareNet.exception import ComputerVisionYolov5Exception

from typing import Optional, List, Literal

import gdown

import zipfile


def download_data_from_drive(url: str, prefix: str):

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


