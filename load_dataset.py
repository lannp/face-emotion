import cv2, os, random, glob

from face_detect import find_faces

emotions = ['neutral', 'anger', 'fear', 'happy', 'sadness', 'surprise']
for emotion in emotions:
    paths = glob.glob("data/%s/*" %(emotion))
    normalized_path = "normalized_images/%s/" %(emotion)
    if not os.path.exists(normalized_path):
        os.makedirs(normalized_path)
    test_path = "test/%s/" %(emotion)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    number_random = int(len(paths) / 10)
    random_paths= random.sample(paths, number_random)

    for path in paths:
        frame = cv2.imread(path)
        normalized_faces = find_faces(frame)
        for face in normalized_faces:
            if path in random_paths:
                cv2.imwrite(test_path+ str(os.path.basename(path)), face[0])
            else:
                cv2.imwrite(normalized_path+ str(os.path.basename(path)), face[0])
