import aug
IMG_SIZE = (480, 480)

IMAGES_DIR = 'test_images'
ANNO_DIR = 'test_labels'
UPLOAD_DIR = 'AUG'
aug.aug_global(IMG_SIZE, IMAGES_DIR, ANNO_DIR, UPLOAD_DIR)
