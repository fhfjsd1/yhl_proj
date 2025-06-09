import os
import pydicom
import numpy as np
from PIL import Image

def dicom_to_png(dicom_path: str, png_path: str) -> None:
    """
    Read a DICOM file and save it as a PNG image after windowing and normalization.
    """
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(np.float32)

    # Apply DICOM rescale if present
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    arr = arr * slope + intercept

    # Normalize to 0-255
    arr -= arr.min()
    arr /= arr.max() if arr.max() != 0 else 1
    arr = (arr * 255).astype(np.uint8)

    Image.fromarray(arr).save(png_path)

def convert_folder(input_dir: str, output_dir: str) -> None:
    """
    Walk through input_dir, convert all .dcm files to .png in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for fname in files:
            dcm_path = os.path.join(root, fname)
            png_fname = os.path.splitext(fname)[0] + '.png'
            png_path = os.path.join(output_dir, png_fname)
            dicom_to_png(dcm_path, png_path)
            print(f"Converted: {dcm_path} -> {png_path}")

if __name__ == "__main__":

    convert_folder("/mnt/2097910f-f006-4c21-8b5a-0815530ed408/ICP/点云数据/CT_分割后", "./CT")