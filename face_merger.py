import cv2
import hashlib
import pandas as pd
from pathlib import Path

from face_cropper import FaceCropper

def main():
    cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cropper = FaceCropper(cascade_file)
    csv_fp = r"D:\paradise\stuff\dreamboothpg\cropped_faces\face_details.csv"
    target_directory_images = Path(r"D:\paradise\stuff\dreamboothpg\swapped_cf")
    save_directory = Path(r"C:\Games\Sacred2")
    cropper.pfb_batch(target_directory_images,save_directory, csv_fp)
    # for img_with_f in target_directory_images.glob('*.jpg'):
    

    # original_image = cv2.imread(str(image_path))
    # modified_image = cropper.put_faces_back(original_image, cropped_faces, csv_file)

    # Display the modified image

if __name__ == '__main__':
    main()
