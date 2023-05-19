from pathlib import Path
import shutil
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from MouthOpen import check_mouth_open
from tqdm import tqdm


app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))



def detect_gender(image_path):
    # Load the image using OpenCV
    img = cv2.imread(str(image_path))

    # Initialize the gender detection model
    # model = insightface.model_zoo.get_model('antelopev2')
    acceptable_size = 200
    # Detect faces in the image
    try:
        faces = app.get(img)
    except:
        return "unknown"

    if len(faces) == 0:
        return "unknown"

    # Get the gender prediction for the first face
    # gender = faces[0].gender

    # Map the gender prediction to 'male' or 'female'
    # breakpoint()
    flag = False
    mouth_open_allfaces = []
    for face in faces:
        # face.bbox[0]
        x1,y1,x2,y2 = face.bbox
        w = x2 - x1
        h = y2 - y1
        if h > acceptable_size and w > acceptable_size:
            flag = True
        mouth_open_allfaces.append(check_mouth_open(face))
    if all(mouth_open_allfaces):
        return 'unknown'
    if not flag:
        return 'unknown'
    gender_str = 'male'
    if any([x.sex for x in faces if x.sex == 'F']):
        return 'female'
    return gender_str


def filter_images_by_gender(directory_path, gender,target_directory = r'C:\temp\deleatb\male_only_pics'):
    # Get the path object for the directory
    dir_path = Path(directory_path)

    # Iterate over the files in the directory
    for file_path in tqdm(dir_path.iterdir(),total=len(list(dir_path.iterdir()))):
        # Check if the file is an image
        if file_path.is_file() and file_path.suffix in ['.jpg', '.jpeg', '.png']:
            # Detect the gender of the image
            image_gender = detect_gender(file_path)

            # Check if the gender matches the specified gender
            target_dir = Path(target_directory) / Path(file_path).parent.name
            if not(image_gender == gender or image_gender == 'unknown'):
                target_dir.mkdir(exist_ok=True,parents=True)

                shutil.move(file_path, target_dir, copy_function=shutil.copy2)
                # print(file_path)

def main(main_dir,result_dir= r'C:\temp\deleatb\male_only_pics'):
    progress = tqdm()
    progress.total = len(list(Path(main_dir).iterdir()))
    for dir in Path(main_dir).iterdir():
        # if dir.is_file() or 'x32gcd' in dir.name: 
        if dir.is_file(): 
            continue
        else:

            existingfp = dir / 'x32gcd.complete'
            if not existingfp.is_file():
                filter_images_by_gender(str(dir), 'male',result_dir)
                existingfp.touch()
        progress.update()
    progress.close()
            # dir.rename(dir.with_name(dir.name+'_x32gcd'))


# Example usage
if __name__ == '__main__':
    # main(r'C:\temp\trial', r'C:\temp\result')
    # main(r'C:\temp\MO_check',r'C:\dumpinggrounds\stable_diff_dg\source')
    main(r'D:\paradise\stuff\new\imageset',r'C:\dumpinggrounds\stable_diff_dg\source')
    main(r'C:\Heaven\Haven\brothel',r'C:\dumpinggrounds\stable_diff_dg\source')
    # main(r'D:\paradise\stuff\new\imageset')
# filter_images_by_gender(r'D:\paradise\stuff\new\imageset\CherryPimps 2022-09-26 Charles Dera Charlotte Sins - Bad Girls Get Bad Grades x207', 'male')