from PIL import Image
from torchvision import transforms, datasets
from imblearn.combine import SMOTETomek

data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def load_and_transform_image(img_path):
    with Image.open(img_path) as img:
        if data_transforms is not None:
            try:
                img = data_transforms(img)
            except:
                print(f'Cannot transform image {img_path}')
        return img
    
def resample(tup):
    img,lab = tup
    images_stacked = list(img)
    label_list_seg = list(lab)
    
    smoteenn = SMOTETomek(sampling_strategy='not majority',random_state = 10)
    resampled_images, resampled_labels = smoteenn.fit_resample(images_stacked, label_list_seg)
    return resampled_images, resampled_labels

def batch_resample(images_stacked,label_list):
    for i in range(images_stacked.shape[0]//1000+1):
        if i != range(images_stacked.shape[0]//1000+1)[-1]:
            yield images_stacked[i*1000:(i+1)*1000],label_list[i*1000:(i+1)*1000]
        elif i == images_stacked.shape[0]//1000:
            yield images_stacked[i*1000:],label_list[i*1000:]
