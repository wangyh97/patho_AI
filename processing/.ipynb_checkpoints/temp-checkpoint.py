from multiprocessing import Pool,set_start_method

def load_and_transform_image(img_path):
    with Image.open(img_path) as img:
        if data_transforms is not None:
            try:
                img = data_transforms(img)
            except:
                print(f'Cannot transform image {img_path}')
        return img

if __name__ == '__main__':
    set_start_method('fork',force=True)
    with Pool(processes=24) as p:
        img_list = list(p.map(load_and_transform_image, tqdm(img_path_list)))
        print('finish one')
