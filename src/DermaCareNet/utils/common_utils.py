import base64
import mimetypes
from pathlib import Path


def image_encode_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()