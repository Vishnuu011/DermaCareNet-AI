import base64
import mimetypes
from pathlib import Path


def image_encode_base64(image_path):

    """
    Encode an image file into a Base64 string.

    Args:
        image_path (str): The path to the image file to encode.

    Returns:
        str: The Base64-encoded string representation of the image.

    Raises:
        FileNotFoundError: If the specified image_path does not exist.
        Exception: For any other unexpected errors during encoding.
    """

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string



def decodeImage(imgstring, fileName):

    """
    Decode a Base64 image string and save it as a binary image file.

    Args:
        imgstring (str): Base64-encoded image string.
        fileName (str): Name of the file (with extension) to save the decoded image as.
                        The file is saved inside the 'data/' directory.

    Returns:
        None

    Raises:
        Exception: If decoding or file writing fails.
    """ 
    
    imgdata = base64.b64decode(imgstring)
    with open("data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()