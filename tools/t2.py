import os
from PIL import Image
import shutil


def move_all_bad_img(root_path):
    print([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    # dirs = ['train']
    ex_folder_list = []
    for dir in dirs:
        img_folders = os.listdir(os.path.join(root_path, dir))
        # img_folders = ['aeroplane']
        for img_folder in img_folders:
            count = 0
            for img in os.listdir(os.path.join(root_path, dir, img_folder)):
                try:
                    im = Image.open(os.path.join(root_path, dir, img_folder, img)).convert('RGB')
                except:
                    ex_folder = os.path.join(root_path, dir, img_folder + '_ex')
                    if not os.path.exists(ex_folder):
                        os.mkdir(ex_folder)
                        ex_folder_list.append(img_folder)
                    shutil.move(os.path.join(root_path, dir, img_folder, img), os.path.join(ex_folder, img))
                    print('损坏图片' + '---' + img)
            print(dir + '-' + img_folder)
        # break
    return ex_folder_list


def remove_single(root_path, dir, img_folder, imgs):
    for img in imgs:
        try:
            im = Image.open(os.path.join(root_path, dir, img_folder, img)).convert('RGB')

        except:
            print('损坏图片' + '---' + img)


if __name__ == '__main__':
    root_path = '/home/zzd/develop/cv/data/domain/visda'
    # dir = 'train'
    # img_folder = 'aeroplane'
    # imgs = ['src_2_02691156_5e29fd83e0494cd815796a932d10290d__180_349_150.png',
    #         'src_1_02691156_5b985bc192961c595de04aad18bd94c3__61_123_165.png',
    #         'src_2_02691156_7b4b249f1d3400e488be2a30dd556a09__44_123_150.png',
    #         'src_2_02691156_1d6afc44b053ab07941d71475449eb25__10_236_150.png',
    #         'src_1_02691156_1e0a24e1135e75a831518807a840c4f4__197_349_165.png',
    #         'src_1_02691156_2a2caad9e540dcc687bf26680c510802__197_10_165.png',
    #         'src_1_02691156_3adbafd59a34d393eccd82bb51193a7f__163_349_165.png',
    #         'src_2_02691156_2a2caad9e540dcc687bf26680c510802__163_10_165.png',
    #         'src_1_02691156_3a3d4a90a2db90b4203936772104a82d__180_349_150.png',
    #         'src_2_02691156_7b39d993f6934a96b08e958a32ef3184__180_10_150.png',
    #         'src_2_02691156_3adbafd59a34d393eccd82bb51193a7f__197_10_165.png',
    #         'src_2_02691156_7b4b249f1d3400e488be2a30dd556a09__316_236_150.png',
    #         'src_1_02691156_4c3b1356008b3284e42e14fe98b0b5__180_349_150.png',
    #         'src_2_02691156_5e6c986605a2817ed4837a4534bf020b__44_349_150.png',
    #         'src_1_02691156_4afcc2cf695baea99a6e43b878d5b335__163_10_165.png',
    #         'src_1_02691156_3b95867a47a8afafe593bc205a118b49__180_349_150.png',
    #         'src_2_02691156_4cb1c851d4333d1c4c3a35cee92bb95b__180_10_150.png',
    #         'src_2_02691156_1e358e70c047687a1a8831797284b67__44_123_150.png'
    #         ]
    # remove_single(root_path, dir, img_folder, imgs)
    ex_folder_list = move_all_bad_img(root_path)
    print(ex_folder_list)
