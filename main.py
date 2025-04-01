import comet_ml
import matplotlib.pyplot as plt
import requests
import torch
from torch import nn
from torchvision import models
import os
from torchcam.methods import SmoothGradCAMpp 
from load_images import build_dataloader
from torchvision.transforms.functional import resize
from torchvision.io.image import read_image

# Configuração dos experimentos
num_classes = 7
image_size = 128
images = [
    "/datasets/FACES_6c/img/test/0/h/109_y__m_h_b.jpg",
    "/datasets/FACES_6c/img/test/0/a/027_o__m_a_a.jpg"
]

# Configuração do Comet ML
comet_api_key = os.getenv('COMET_API_KEY')
experiment_key = "f7fb8be812dc4489ae96b4583bc70d5c"


COMET_BASE_URL = "https://www.comet.com/api/rest/v2"
HEADERS = {"Authorization": comet_api_key, "Content-Type": "application/json"}

asset_id = 'adebef4600144192bec7c810b3e3d809'
# Baixar o modelo do Comet ML
model_path = "model.pth"
response = requests.get(f"{COMET_BASE_URL}/experiment/asset/get-asset?experimentKey={experiment_key}&assetId={asset_id}", headers=HEADERS)
if response.status_code == 200:
    with open(model_path, "wb") as f:
        f.write(response.content)
else:
    print(f'status code {response.status_code} - {response.text}')
    raise Exception("Erro ao baixar o modelo do Comet ML")


#model = models.resnet18(pretrained=False)
#model.load_state_dict(torch.load(model_path))
#model.eval()
dl = build_dataloader(image_size=image_size, images=images)
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
model.classifier.fc = nn.Linear(1280, num_classes)
model.load_state_dict(torch.load(model_path)['model_state_dict'])

img = read_image(images[0])
input_tensor = resize(img,  (image_size, image_size))
cam_extractor = SmoothGradCAMpp(model)
print(input_tensor.unsqueeze(0).shape)
out = model(input_tensor.unsqueeze(0).float())


activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

plt.imsave(activation_map[0].squeeze(0).numpy());
print("Processo concluído e imagens enviadas para o Comet ML!")
