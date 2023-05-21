import cv2
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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
        h_fac = 0.30
        # breakpoint()
        faces = sorted(faces, key=lambda x: x[2] * x [3], reverse= True)
        for i, (x1, y1, w1, h1) in enumerate(faces):
            if w1 * h1 < 70 * 70:
                break
            y = max(0, y1 - int(h1 * h_fac))
            h = h1 + int(h1 * (h_fac + 0.10))
            increase_pixel = h - w1
            w_fac = int(increase_pixel / 2)
            x = max(0, x1 - w_fac)
            w = w1 + w_fac
            face = image[y:y+h, x:x+w]
            # face = cv2.resize(face, (512, 512))
            sub_dir = save_directory / ip.parent.name
            sub_dir.mkdir(parents=True, exist_ok=True)
            save_path = sub_dir / ip.name
            cv2.imwrite(str(save_path), face)
            cropped_images.append(face)

            face_id = i
            face_info = {
                'uni_name': f'Yummyx (17) @hudengi {ip.parent.name} W1t81N {ip.name}',
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
        rows_to_delete = []
        for index, row in tqdm(df.iterrows()):
            face_path =  str(target_directory_images / row['uni_name'])
            original_image = row['source_image']
            x = row['x']
            y = row['y']
            w = row['width']
            h = row['height']
            done = self.put_faces_back(original_image, face_path, (x,y,w,h),result_dir,csv_file)
            if done:
                rows_to_delete.append(index)
        df = df.drop(rows_to_delete)
        df.to_csv(csv_file, index=False)

    def put_faces_back(self, original_image_p, face_images_p, face_box, result_dir,csv_file):
        original_image = cv2.imread(original_image_p)

        fip = Path(face_images_p)
        flag = False
        for i in range(1, 5):
            copy_image = original_image.copy()
            rip = fip.with_stem(fip.stem+'_'+str(i))
            if not rip.is_file(): 
                continue

            # breakpoint()
            flag = True
            face_images = cv2.imread(str(rip))
            x, y, w, h = face_box
            face_resized = cv2.resize(face_images, (w, h))
            copy_image[y:y+h, x:x+w] = face_resized
            cv2.imwrite(str(result_dir / rip.name), copy_image)
            rip.unlink()
        return flag

        return original_image

def doit_dir(target_dir):
    
    cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cropper = FaceCropper(cascade_file)
    # target_directory_images = Path(r"D:\paradise\stuff\essence\Pictures\HeapOfHoors\champions")
    target_directory_images = Path(target_dir)
    save_directory = Path(r"D:\paradise\stuff\dreamboothpg\cropped_faces")
    for img_with_f in tqdm(target_directory_images.glob('*.jpg')):
        # breakpoint()
        # try:
            cropper.crop_faces(img_with_f, save_directory)
        # except:

            # continue
def main():
    target_parent_dir = Path(r'C:\Heaven\Haven\brothel')
    for dir in target_parent_dir.iterdir():
        if dir.is_dir():
            doit_dir(str(dir))


if __name__ == '__main__':
    main()
