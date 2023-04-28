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

# Load the image
def check_image(fp):
    image = cv2.imread(fp)

    # Load the model


    # Detect faces in the image
    faces = app.get(image)

    # Check if each face is covered
    result = []
    for face in faces:
        result.append(check_face_coverage(face))
    if len(result) <= 0:
        return False
    return all(result)

def main():
    parent_dir = Path(r'C:\temp\deletable')
    for files in parent_dir.glob('*.jpg'):
        if check_image(str(files)):
            shutil.copy(files, r'C:\temp\deleatble')



if __name__=='__main__':
    main()