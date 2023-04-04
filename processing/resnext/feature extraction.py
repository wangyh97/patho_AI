import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

resnet50 = models.resnet50(pretrained=True)
modules=list(resnet50.children())[:-1]
resnet50=nn.Sequential(*modules)

def _get_features(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img_data = transform(img)
    img_data = img_data.unsqueeze(0)
    resnet_features = resnet50(img_data)
    return resnet_features.detach().numpy()

def get_patient_features(uuid):
    
