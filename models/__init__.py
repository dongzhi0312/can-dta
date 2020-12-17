import torch

# from datasets import DATA_SETS
from models.visda_architectures import create_resnet_lower, create_resnet_upper


def create_feature_extractor(args):
    """
    获取网络的前几层作为特征提取网络
    :param args:
    :return:
    """
    encoder = create_resnet_lower(args.model)

    if args.encoder_ckpt_path:
        print('load encoder from {}'.format(args.encoder_ckpt_path))
        ckpt = torch.load(args.encoder_ckpt_path)
        encoder.load_state_dict(ckpt['feature_extractor_state_dict'])
        return encoder
    return encoder


def create_classifier(args):
    """
    获取网络的后几层作为分类器
    :param args:
    :return:
    """
    # TODO
    # DATA_SETS = {}
    # num_classes = DATA_SETS[args.source_dataset_code].num_class()
    num_classes = 12
    class_classifier = create_resnet_upper(args.model, num_classes=num_classes)

    if args.classifier_ckpt_path:
        print('load class classifier from {}'.format(args.classifier_ckpt_path))
        ckpt = torch.load(args.classifier_ckpt_path)
        class_classifier.load_state_dict(ckpt['classifier_state_dict'])
        return class_classifier
    return class_classifier

# import torch
#
# from datasets import DATA_SETS
# from models.visda_architectures import create_resnet_lower, create_resnet_upper
#
#
# def create_class_classifier(args):
#     num_classes = DATA_SETS[args.source_dataset_code].num_class()
#
#     class_classifier = create_resnet_upper(args.model, num_classes=num_classes)
#
#     if args.classifier_ckpt_path:
#         print("Load class classifier from {}".format(args.classifier_ckpt_path))
#         ckpt = torch.load(args.classifier_ckpt_path)
#         class_classifier.load_state_dict(ckpt['classifier_state_dict'])
#         return class_classifier
#     return class_classifier
#
#
# def create_feature_extractor(args):
#     encoder = create_resnet_lower(args.model)
#
#     if args.encoder_ckpt_path:
#         print("Load encoder from {}".format(args.encoder_ckpt_path))
#         ckpt = torch.load(args.encoder_ckpt_path)
#         encoder.load_state_dict(ckpt['feature_extractor_state_dict'])
#         return encoder
#     return encoder
