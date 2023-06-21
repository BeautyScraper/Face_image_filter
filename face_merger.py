import cv2
import hashlib
import pandas as pd
from pathlib import Path

from face_cropper import FaceCropper

def main():
    cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    # csv_fp = r"D:\paradise\stuff\dreamboothpg\cropped_faces\face_details.csv"
    # target_directory_images = Path(r"D:\paradise\stuff\dreamboothpg\swapped_cf")
    csv_fp = r"C:\dumpinGGrounds\cropped_faces\face_details.csv"
    target_directory_images = Path(r"C:\dumpinGGrounds\results-20230615T080448Z-001\results")
    save_directory = Path(r"C:\Personal\Games\Sacred2")
    cropper = FaceCropper(r"C:\temp")
    # save_directory = Path(r"D:\paradise\stuff\dreamboothpg\res")
    cropper.pfb_batch(target_directory_images,save_directory, csv_fp)
    # for img_with_f in target_directory_images.glob('*.jpg'):
    

    # original_image = cv2.imread(str(image_path))
    # modified_image = cropper.put_faces_back(original_image, cropped_faces, csv_file)

    # Display the modified image

if __name__ == '__main__':
    main()
