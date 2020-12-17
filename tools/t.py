import os
from PIL import Image

if __name__ == '__main__':
    # feats = []
    # [feats[key] for key in feats if key in ['prob', 'feat']]
    root_path = '/home/zzd/develop/cv/data/domain/visda'
    # lists = os.listdir(root_path)
    # for l in lists:
    #     # print(l)
    #     if os.path.isdir(os.path.join(root_path, l)):
    #         print(l)
    print([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    # dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    dirs = ['train']
    for dir in dirs:
        img_folders = os.listdir(os.path.join(root_path, dir))
        for img_folder in img_folders:
            count = 0
            for img in os.listdir(os.path.join(root_path, dir, img_folder)):
                try:
                    im = Image.open(os.path.join(root_path, dir, img_folder, img)).convert('RGB')
                except:
                    print('图片损坏' + img)
            print(dir + '-' + img_folder)
        break
