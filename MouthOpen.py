import shutil
import numpy as np
import insightface
import cv2
from insightface.app import FaceAnalysis
from pathlib import Path



app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def check_face_coverage(face):
    # Check if the nose is covered by an object
    # breakpoint()
    # landmarks = face.landmark.astype(np.int)
    landmarks = face['landmark_3d_68'].astype(np.int)
    if landmarks[30][1] < 0:
        return True
    else:
        return False

def check_mouth_open(face):
    landmarks = face['landmark_3d_68']

    # Calculate scaled distance between upper and lower lip landmarks
    # breakpoint()
    mouth_height = np.linalg.norm(landmarks[66] - landmarks[62])
    nose_distance = np.linalg.norm(landmarks[29] - landmarks[28])
    # nose_distance = landmarks[7][1] - landmarks[2][1]
    scaled_distance = mouth_height / nose_distance

    # Check if mouth is open
    if scaled_distance > 1.5:
        return True


# Load the image
def check_image(fp):
    image = cv2.imread(fp)

    # Load the model


    # Detect faces in the image
    faces = app.get(image)

    # Check if each face is covered
    result = []
    for face in faces:
        result.append(check_mouth_open(face))
    if len(result) <= 0:
        return False
    return all(result)

def main():
    parent_dir = Path(r'D:\paradise\stuff\new\imageset\TeamSkeet Wendy Moon - Political Plower 84x 1620x1080 10-20-2022')
    for files in parent_dir.glob('*.jpg'):
        print(files)
        if check_image(str(files)):
            shutil.copy(files, r'C:\temp\deleatble')



if __name__=='__main__':
    main()