import cv2
import hashlib
import pandas as pd
from pathlib import Path

class FaceCropper:
    def __init__(self, cascade_file):
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def crop_faces(self, image_path, save_directory):
        image = cv2.imread(str(image_path))
        faces = self.detect_faces(image)

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        ip = Path(image_path)

        csv_file = save_directory / 'face_details.csv'

        cropped_images = []
        face_details = []
        
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]
            sub_dir = save_directory / ip.parent.name
            sub_dir.mkdir(parents=True, exist_ok=True)
            save_path = sub_dir / ip.name
            cv2.imwrite(str(save_path), face)
            cropped_images.append(face)

            face_id = i
            face_info = {
                'uni_name': f'Yummyx (17) @hudengi {ip.parent.name} W1t81N {ip.stem}',
                'source_image': image_path,
                'x': x,
                'y': y,
                'width': w,
                'height': h
            }
            face_details.append(face_info)
            break

        # Save face details to a CSV file
        if csv_file.exists():
            existing_df = pd.read_csv(csv_file)
            face_details_df = pd.DataFrame(face_details)
            updated_df = pd.concat([existing_df, face_details_df], ignore_index=True)
            updated_df.to_csv(csv_file, index=False)
        else:
            df = pd.DataFrame(face_details)
            df.to_csv(csv_file, index=False)
        # return cropped_images, face_details
        # return cropped_images, face_details

    def pfb_batch(self,target_directory_images,result_dir,csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            breakpoint()

    def put_faces_back(self, original_image, face_images, face_box):
        face_positions = pd.read_csv(csv_file)

        for _, row in face_positions.iterrows():
            x, y, w, h = face_box
            
            for face in face_images:
                face_resized = cv2.resize(face, (w, h))
                original_image[y:y+h, x:x+w] = face_resized

        return original_image

def main():
    cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cropper = FaceCropper(cascade_file)
    target_directory_images = Path(r"D:\paradise\stuff\essence\Pictures\HeapOfHoors\champions")
    save_directory = Path(r"D:\paradise\stuff\dreamboothpg\cropped_faces")
    for img_with_f in target_directory_images.glob('*.jpg'):
        try:
            cropper.crop_faces(img_with_f, save_directory)
        except:
            continue
    

    # original_image = cv2.imread(str(image_path))
    # modified_image = cropper.put_faces_back(original_image, cropped_faces, csv_file)

    # Display the modified image

if __name__ == '__main__':
    main()
