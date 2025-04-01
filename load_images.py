import torchvision.transforms as transforms
import numpy as np
import dlib
from PIL import Image
from torch.utils.data import DataLoader, Dataset

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

def get_facial_landmarks(image, load_image=True) -> np.ndarray[68, 2]:
        if load_image:
            image = dlib.load_rgb_image(image)
        dets = detector(image, 1)
        for k, d in enumerate(dets):
            shape = predictor(image, d)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            return landmarks, image
        return np.zeros((68, 2)), image

class MiniFacesDataset(Dataset):

    def __init__(self, transforms, images, landmarks = False):
        self.landmarks = landmarks
        self.transforms = transforms
        self.images = images
        self.faces_numbers = {"a": 0, "d": 1, "f": 2, "h": 3, "n": 4, "s": 5}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        transformed_image = self.transforms(image)
        label = image_path.split('/')[::-1][1]

        return transformed_image, self.faces_numbers[label]

def build_dataloader(image_size: int, images: list[str]):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return MiniFacesDataset(images=images, transforms=transform)


if __name__ == "__main__":
    # used to test the script
    image_size = 128
    images = [
        "/datasets/FACES_6c/img/test/0/h/109_y__m_h_b.jpg",
        "/datasets/FACES_6c/img/test/0/a/027_o__m_a_a.jpg"
    ]
    dl = build_dataloader(image_size=image_size, images=images)
    img, label = next(iter(dl))
    print(img.shape)
    print(label)
