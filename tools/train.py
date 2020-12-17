import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
import torch.optim as optim
# from model import model
from config.config import cfg, cfg_from_file, cfg_from_list
# from .prepare_data import *
from prepare_data import *
from models import create_feature_extractor, create_classifier
import sys
import pprint


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--method', dest='method',
                        help='set the method to use',
                        default='CAN', type=str)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name',
                        default='exp', type=str)
    parser.add_argument('--model', dest='model',
                        help='the model name',
                        default='resnet50', type=str)
    parser.add_argument('--encoder_ckpt_path', dest='encoder_ckpt_path',
                        help='encoder_ckpt_path',
                        default='', type=str)
    parser.add_argument('--classifier_ckpt_path', dest='classifier_ckpt_path',
                        help='classifier_ckpt_path',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def train(args):
    bn_domain_map = {}

    # method-specific setting 
    if args.method == 'CAN':
        from solver.can_solver import CANSolver as Solver
        dataloaders = prepare_data_CAN()
        num_domains_bn = 2

    elif args.method == 'MMD':
        from solver.mmd_solver import MMDSolver as Solver
        dataloaders = prepare_data_MMD()
        num_domains_bn = 2

    elif args.method == 'SingleDomainSource':
        from solver.single_domain_solver import SingleDomainSolver as Solver
        dataloaders = prepare_data_SingleDomainSource()
        num_domains_bn = 1

    elif args.method == 'SingleDomainTarget':
        from solver.single_domain_solver import SingleDomainSolver as Solver
        dataloaders = prepare_data_SingleDomainTarget()
        num_domains_bn = 1

    else:
        raise NotImplementedError("Currently don't support the specified method: %s."
                                  % args.method)

    # initialize model
    model_state_dict = None
    fx_pretrained = True
    resume_dict = None

    if cfg.RESUME != '':
        resume_dict = torch.load(cfg.RESUME)
        model_state_dict = resume_dict['model_state_dict']
        fx_pretrained = False
    elif cfg.WEIGHTS != '':
        param_dict = torch.load(cfg.WEIGHTS)
        model_state_dict = param_dict['weights']
        bn_domain_map = param_dict['bn_domain_map']
        fx_pretrained = False

    # net = model.danet(num_classes=cfg.DATASET.NUM_CLASSES,
    #                   state_dict=model_state_dict,
    #                   feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR,
    #                   frozen=[cfg.TRAIN.STOP_GRAD],
    #                   fx_pretrained=fx_pretrained,
    #                   dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
    #                   fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
    #                   num_domains_bn=num_domains_bn)


    # 用 dta 的 feature_ectractor 和 classifier 代替 can 的 DANet 创建特征提取器和分类器
    feature_extractor, classifier = create_feature_extractor(args), create_classifier(args)
    models = {
        'feature_extractor': feature_extractor,
        'classifier': classifier
    }

    # net = torch.nn.DataParallel(net)  # 多卡并行
    # if torch.cuda.is_available():
    #     net.cuda()

    models['feature_extractor'] = torch.nn.DataParallel(models['feature_extractor'])  # 多卡并行
    models['classifier'] = torch.nn.DataParallel(models['classifier'])  # 多卡并行
    if torch.cuda.is_available():
        models['feature_extractor'].cuda()
        models['classifier'].cuda()

    def _create_optimizers(args, feature_extractor, classifier):
        if args.optimizer == 'Adam':
            return {
                'feature_extractor': optim.Adam(feature_extractor.parameters(), lr=args.lr,
                                                weight_decay=args.weight_decay),
                'classifier': optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay),
            }

        return {
            'feature_extractor': optim.SGD(feature_extractor.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                           momentum=args.momentum),
            'classifier': optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum),
        }

    # initialize solver
    # train_solver = Solver(net, dataloaders, bn_domain_map=bn_domain_map, resume=resume_dict)
    # dta fe clr 代替 danet
    train_solver = Solver(models, dataloaders, bn_domain_map=bn_domain_map, resume=resume_dict)

    # train 
    train_solver.solve()
    print('Finished!')


if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.resume is not None:
        cfg.RESUME = args.resume
    if args.weights is not None:
        cfg.MODEL = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    train(args)
