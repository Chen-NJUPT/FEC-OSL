from __future__ import division
__author__ = 'HPC'
import sys
import os
import argparse

base_dir = os.path.dirname(os.path.abspath(__file__))
byte_extract_path = os.path.join(base_dir, 'byte_features_extract')
sys.path.append(byte_extract_path)
flow_extract_path = os.path.join(base_dir, 'flow_features_extract')
sys.path.append(flow_extract_path)
energy_model_path = os.path.join(base_dir, 'energy_model')
sys.path.append(energy_model_path)

from byte_features_extract import header_dataset_generation, payload_dataset_generation, data_process_header, data_process_payload
from flow_features_extract import flow_main_model

from FEC_OSL_model import FECOSLModel
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from timm.data.mixup import Mixup
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import byte_features_extract.header_dataset_generation as header_dataset_generation
import byte_features_extract.payload_dataset_generation as payload_dataset_generation
import h5py
import energy_model.energy as energy
from scipy.stats import weibull_min
import dgl
from torch.cuda.amp import GradScaler
from sklearn.metrics import adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment

def get_classes(path):
    items = os.listdir(path)
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            file_count = len([name for name in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, name))])
    nb_classes = len(items)
    return nb_classes

def get_args_parser_byte():
    parser = argparse.ArgumentParser('byte fine-tuning for traffic classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='FEC_OSL_byte_header_TransFormer', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--finetune', 
                        help='finetune from checkpoint')
    parser.add_argument('--data_path', default="dataset/byte_data/tor_session/tor_header", type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='dataset/output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def get_byte_header_dataset(dataset_header_path, ratio, new_classes):
    args = get_args_parser_byte()
    args = args.parse_args()
    args.data_path = dataset_header_path
    args.ratio = ratio
    args.new_classes = new_classes
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    dataset_train, dataset_val, unknow_dataset_train, unknow_dataset_val = header_dataset_generation.build_dataset(args=args)
    sampler_know_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_know_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_unknow_train = torch.utils.data.SequentialSampler(unknow_dataset_train)
    sampler_unknow_val = torch.utils.data.SequentialSampler(unknow_dataset_val)
    data_loader_train_know = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_know_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val_know = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_know_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_train_unknow = torch.utils.data.DataLoader(
        unknow_dataset_train, sampler=sampler_unknow_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    data_loader_val_unknow = torch.utils.data.DataLoader(
        unknow_dataset_val, sampler=sampler_unknow_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_train_know, data_loader_val_know, data_loader_train_unknow, data_loader_val_unknow

def get_byte_payload_dataset(dataset_payload_path, ratio, new_classes):
    args = get_args_parser_byte()
    args = args.parse_args()
    args.data_path = dataset_payload_path
    device = torch.device(args.device)
    args.ratio = ratio
    args.new_classes = new_classes
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    dataset_train, dataset_val, unknow_dataset_train, unknow_dataset_val = payload_dataset_generation.build_dataset(args=args)
    sampler_know_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_know_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_unknow_train = torch.utils.data.SequentialSampler(unknow_dataset_train)
    sampler_unknow_val = torch.utils.data.SequentialSampler(unknow_dataset_val)
    data_loader_train_know = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_know_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val_know = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_know_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_train_unknow = torch.utils.data.DataLoader(
        unknow_dataset_train, sampler=sampler_unknow_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    data_loader_val_unknow = torch.utils.data.DataLoader(
        unknow_dataset_val, sampler=sampler_unknow_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_train_know, data_loader_val_know, data_loader_train_unknow, data_loader_val_unknow
    
def get_flow_dataset(dataset_flow_path, all_classes, known_classes):
    data_loader_flow_train_known = flow_main_model.model(dataset_flow_path, randseed=256, splitrate=0.1,
        all_classes = all_classes, known_classes = known_classes).parse_raw_data(all_classes, known_classes)
    data_loader_flow_valid_known = flow_main_model.model(dataset_flow_path, randseed=256, splitrate=0.1,
        all_classes = all_classes, known_classes = known_classes).parse_raw_data(all_classes, known_classes)
    data_loader_flow_train_aunknown = flow_main_model.model(dataset_flow_path, randseed=256, splitrate=0.1,
        all_classes = all_classes, known_classes = known_classes).parse_raw_data(all_classes, known_classes)
    data_loader_flow_valid_unknown = flow_main_model.model(dataset_flow_path, randseed=256, splitrate=0.1,
        all_classes = all_classes, known_classes = known_classes).parse_raw_data(all_classes, known_classes)
    return data_loader_flow_train_known, data_loader_flow_valid_known, \
        data_loader_flow_train_aunknown, data_loader_flow_valid_unknown

def save_dataloader_to_h5(data_loader, h5_filename):
    if os.path.exists(h5_filename):
        return
    base_dir = os.path.join("dataset", "dataloader")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with h5py.File(h5_filename, 'w') as h5f:
        dataset_size = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        first_batch = next(iter(data_loader))
        features_shape = first_batch[0].shape[1:]
        labels_shape = first_batch[1].shape[1:]

        features_dataset = h5f.create_dataset('features', shape=(dataset_size,) + features_shape, dtype='float32')
        labels_dataset = h5f.create_dataset('labels', shape=(dataset_size,) + labels_shape, dtype='int64')

        idx = 0
        for batch_idx, (features, labels) in enumerate(data_loader):
            current_batch_size = features.size(0)

            features_np = features.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            features_dataset[idx:idx + current_batch_size] = features_np
            labels_dataset[idx:idx + current_batch_size] = labels_np

            idx += current_batch_size

        print(f"Data successfully saved to {h5_filename}")

def train(model, nb_classes, use_gpu, device, dataset_name, 
        data_loader_byte_header_train_know, data_loader_byte_header_val_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_val_know,
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown,
        ratio):
    torch.manual_seed(256)
    np.random.seed(256)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cudnn.benchmark = True  
    model.to(device)
    model.byte_header_extractor.to(device)
    model.byte_payload_extractor.to(device)
    model.flow_extractor.to(device)
    model.energy_classifier.to(device)
    model.odc_classifier.to(device)
    model.train(True)
    model.byte_header_extractor.train(True)
    model.byte_payload_extractor.train(True)
    model.flow_extractor.train(True)
    model.energy_classifier.train(True)
    model.odc_classifier.train(True)
    model.odc_classifier.init_weights()

    if os.path.exists('dataset/saved_models/' + dataset_name + '_FEC_OSL_model_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth'):
        
        model = torch.load('dataset/saved_models/' + dataset_name + '_FEC_OSL_model_' + 
                str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth')
    else:
        print("Start Cross-iteration")
        train_ft(model, nb_classes, use_gpu, device, dataset_name,
            data_loader_byte_header_train_know, data_loader_byte_header_val_know,
            data_loader_byte_payload_train_know, data_loader_byte_payload_val_know,
            data_loader_flow_train_known, data_loader_flow_valid_known,
            data_loader_byte_header_train_unknow,
            data_loader_byte_payload_train_unknow,
            data_loader_flow_train_unknown, ratio)
    return model

def train_ft(model, nb_classes, use_gpu, device, dataset_name, 
        data_loader_byte_header_train_know, data_loader_byte_header_val_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_val_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown,
        ratio):
    for param in model.byte_header_extractor.parameters():
        param.requires_grad = True
    for param in model.byte_payload_extractor.parameters():
        param.requires_grad = True
    for param in model.flow_extractor.parameters():
        param.requires_grad = True
    for param in model.energy_classifier.parameters():
        param.requires_grad = True
    for param in model.odc_classifier.parameters():
        param.requires_grad = True
    
    args = get_args_parser_byte()
    args = args.parse_args()
    args.nb_classes = int(nb_classes * ratio)
    args.epoch = 50
    seed = args.seed + misc.get_rank()  
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
    model_without_ddp_header = model.byte_header_extractor
    model_without_ddp_payload = model.byte_payload_extractor
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  
        args.lr = args.blr * eff_batch_size / 256
        
    param_groups_header = lrd.param_groups_lrd(model_without_ddp_header, args.weight_decay,
        no_weight_decay_list=model_without_ddp_header.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    param_groups_payload = lrd.param_groups_lrd(model_without_ddp_payload, args.weight_decay,
        no_weight_decay_list=model_without_ddp_payload.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer_byte_header = torch.optim.SGD(param_groups_header, lr=1e-4, momentum=0.9)
    optimizer_byte_payload = torch.optim.SGD(param_groups_payload, lr=1e-4, momentum=0.9)
    optimizer_flow = torch.optim.SGD(model.flow_extractor.parameters(), lr=1e-4, momentum=0.9)
    
    scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flow, mode='min', factor=0.5, patience=3, verbose=True)
    optimizer_energy = torch.optim.SGD(model.energy_classifier.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler_energy = torch.optim.lr_scheduler.LambdaLR(
        optimizer_energy,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epoch * len(data_loader_byte_header_train_know),
            1,
            1e-6 / 1e-3))
    
    optimizer_odc_classifier = torch.optim.SGD(model.odc_classifier.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler_odc_classifier = torch.optim.lr_scheduler.LambdaLR(
        optimizer_odc_classifier,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epoch * len(data_loader_byte_header_train_unknow),
            1,
            1e-6 / 1e-3))
    
    scaler = GradScaler()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    T = 10
    save_dataloader_to_h5(data_loader_byte_header_train_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_train_know_' + str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5')
    save_dataloader_to_h5(data_loader_byte_header_train_unknow, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_train_unknow_' + str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5')
    
    save_dataloader_to_h5(data_loader_byte_payload_train_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_train_know_' + str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5')
    save_dataloader_to_h5(data_loader_byte_payload_train_unknow, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_train_unknow_' + str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5')

    graphs_train_known = []
    labels_train_known = []
    graphs_train_unknown = []
    labels_train_unknown = []

    for i in data_loader_flow_train_known.know_train_index:
        graphs_train_known.append(data_loader_flow_train_known.graphs[i])
        labels_train_known.append(data_loader_flow_train_known.labelId[i])
    for i in data_loader_flow_train_unknown.unknow_train_index:
        graphs_train_unknown.append(data_loader_flow_train_unknown.graphs[i])
        labels_train_unknown.append(data_loader_flow_train_unknown.labelId[i])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_know_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_header_train_know = torch.tensor(h5f['features'])
        label_byte_header_train_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_know_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_payload_train_know = torch.tensor(h5f['features'])
        label_byte_payload_train_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_unknow_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_header_train_unknow = torch.tensor(h5f['features'])
        label_byte_header_train_unknow = torch.tensor(h5f['labels'])  
        label_byte_header_train_unknow[:] = int(nb_classes * ratio)
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_unknow_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_payload_train_unknow = torch.tensor(h5f['features']) 
        label_byte_payload_train_unknow = torch.tensor(h5f['labels'])
        label_byte_payload_train_unknow[:] = int(nb_classes * ratio)

    
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_unknow_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        true_unknow_labels = torch.tensor(h5f['labels'])
        
    with torch.no_grad():
        
        features_all, outputs_all = model(feature_byte_header_train_unknow.to(device), feature_byte_payload_train_unknow.to(device), dgl.batch(graphs_train_unknown).to(device), 'odc')
        
        pred_labels = model.odc_classifier.initialize_memory_bank(features_all, num_clusters=nb_classes - int(nb_classes * ratio)) 
        pred_labels_numpy = pred_labels.cpu().numpy()
        true_unknow_labels_numpy = true_unknow_labels.cpu().numpy()
        ami_score = adjusted_mutual_info_score(true_unknow_labels_numpy, pred_labels_numpy)
        
    feature_byte_header_train, label_byte_header_train = \
        merge_and_shuffle([feature_byte_header_train_know, feature_byte_header_train_unknow],
                        [label_byte_header_train_know, label_byte_header_train_unknow])
    feature_byte_payload_train, label_byte_payload_train = \
        merge_and_shuffle([feature_byte_payload_train_know, feature_byte_payload_train_unknow],
                        [label_byte_payload_train_know, label_byte_payload_train_unknow])
        
    unknow_len = len(labels_train_unknown)
    know_len = len(labels_train_known)
    labels_train_unknown = [int(nb_classes * ratio) for i in range(unknow_len)]  
    flow_graphs_train, flow_labels_train = merge_and_shuffle_graphs(
        graphs_train_known + graphs_train_unknown,
        labels_train_known +  labels_train_unknown
    )

    for i in range(args.epoch):
        iterator_know_header = iter(data_loader_byte_header_train_know)
        iterator_know_payload = iter(data_loader_byte_payload_train_know)
        iterator_unknow_header = iter(data_loader_byte_header_train_unknow)
        iterator_unknow_payload = iter(data_loader_byte_payload_train_unknow)
        
        data_loader_flow_train_unknown.reflesh()  
        data_loader_flow_train_known.reflesh()
        
        optimizer_byte_header.zero_grad()
        optimizer_byte_payload.zero_grad()
        optimizer_flow.zero_grad()
        optimizer_energy.zero_grad()
        print(str(i + 1) + "epoch:")
        print(f"forward: Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"forward: Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        batch_size = args.batch_size
        total_size = label_byte_header_train.shape[0]  
        batch_count = math.ceil(total_size / args.batch_size)
        pred_all = []
        target_all = []
        sigh_1 = 0  
        sigh_2 = 0  
        sigh_3 = 0  
        total_loss = 0
        cluster_labels = []
        true_unknown_labels_list = []
        features_list = []
        for j in range(0, total_size, batch_size):
            end = j + batch_size
            end_1 = sigh_1 + batch_size
            end_2 = sigh_2 + batch_size
            end_3 = sigh_3 + batch_size
            try:
                samples_header_train_know, targets_header_train_know = next(iterator_know_header)
                samples_payload_train_know, targets_payload_train_know = next(iterator_know_payload)
            except StopIteration:
                iterator_know_header = iter(data_loader_byte_header_train_know)
                iterator_know_payload = iter(data_loader_byte_payload_train_know)
                samples_header_train_know, targets_header_train_know = next(iterator_know_header)
                samples_payload_train_know, targets_payload_train_know = next(iterator_know_payload)
                
            try:
                samples_header_train_unknow, targets_header_train_unknow = next(iterator_unknow_header)
                samples_payload_train_unknow, targets_payload_train_unknow = next(iterator_unknow_payload)
                
            except StopIteration:
                iterator_unknow_header = iter(data_loader_byte_header_train_unknow)
                iterator_unknow_payload = iter(data_loader_byte_payload_train_unknow)
                samples_header_train_unknow, targets_header_train_unknow = next(iterator_unknow_header)
                samples_payload_train_unknow, targets_payload_train_unknow = next(iterator_unknow_payload)
            
            samples_header_train = feature_byte_header_train[j:end]
            targets_header_train = label_byte_header_train[j:end]
            samples_payload_train = feature_byte_payload_train[j:end]
            targets_payload_train = label_byte_header_train[j:end]
            
            graphs_train = dgl.batch(flow_graphs_train[j:end]).to(device)
            labels_train = flow_labels_train[j:end]
            if (sigh_1 % know_len) < (end_1 % know_len):
                graphs_train_know = dgl.batch(graphs_train_known[(sigh_1 % know_len):(end_1 % know_len)]).to(device)
                labels_train_know = torch.tensor(labels_train_known[(sigh_1 % know_len):(end_1 % know_len)])
                sigh_1 = end_1
            else:
                
                graphs_train_know = dgl.batch(graphs_train_known[(sigh_1 % know_len):know_len]).to(device)
                labels_train_know = torch.tensor(labels_train_known[(sigh_1 % know_len):know_len])
                sigh_1 = 0
            if (sigh_2 % unknow_len) < (end_2 % unknow_len):
                graphs_train_unknow = dgl.batch(graphs_train_unknown[(sigh_2 % unknow_len):(end_2 % unknow_len)]).to(device)
                labels_train_unknow = torch.tensor(labels_train_unknown[(sigh_2 % unknow_len):(end_2 % unknow_len)])
                sigh_2 = end_2
            else:
                graphs_train_unknow = dgl.batch(graphs_train_unknown[(sigh_2 % unknow_len):unknow_len]).to(device)
                labels_train_unknow = torch.tensor(labels_train_unknown[(sigh_2 % unknow_len):unknow_len])
                sigh_2 = 0
                
            if use_gpu :
                graphs_train = graphs_train.to(torch.device(device))
                labels_train = labels_train.to(torch.device(device))
                graphs_train_know = graphs_train_know.to(torch.device(device))
                labels_train_know = labels_train_know.to(torch.device(device))
                graphs_train_unknow = graphs_train_unknow.to(torch.device(device))
                labels_train_unknow = labels_train_unknow.to(torch.device(device))
                
            if mixup_fn is not None:
                samples_header_train, targets_header_train = mixup_fn(samples_header_train, targets_header_train)
                samples_payload_train, targets_payload_train = mixup_fn(samples_payload_train, targets_payload_train)
                samples_header_train_know, targets_header_train_know = mixup_fn(samples_header_train_know, targets_header_train_know)
                samples_payload_train_know, targets_payload_train_know = mixup_fn(samples_payload_train_know, targets_payload_train_know)
                samples_header_train_unknow, targets_header_train_unknow = mixup_fn(samples_header_train_unknow, targets_header_train_unknow)
                samples_payload_train_unknow, targets_payload_train_unknow = mixup_fn(samples_payload_train_unknow, targets_payload_train_unknow)
                
            with torch.cuda.amp.autocast(enabled=False):
                features_all, outputs = model(samples_header_train.to(device), samples_payload_train.to(device), graphs_train, 'energy')
                features_in, outputs_in = model(samples_header_train_know.to(device), samples_payload_train_know.to(device), graphs_train_know, 'energy')
                features, outputs_out = model(samples_header_train_unknow.to(device), samples_payload_train_unknow.to(device), graphs_train_unknow, 'energy')
                labels = labels_train
                
            loss = energy.energy_ft_loss(outputs, outputs_in, outputs_out, labels) 
            
            if (sigh_3 % unknow_len) < (end_3 % unknow_len):
                idx = torch.arange((sigh_3 % unknow_len), (end_3 % unknow_len))
                true_unknown_labels_list.append(true_unknow_labels[(sigh_3 % unknow_len):(end_3 % unknow_len)])
                sigh_3 = end_3
            else:
                idx = torch.arange((sigh_3 % unknow_len), unknow_len)
                true_unknown_labels_list.append(true_unknow_labels[(sigh_3 % unknow_len):unknow_len])
                sigh_3 = 0
                
            cls_score = model.odc_classifier(features)
            
            pseudo_labels = model.odc_classifier.memory_bank['labels'][idx].clone().detach().to(dtype=torch.long).to(cls_score[0].device)
            losses = model.odc_classifier.loss(features, model.odc_classifier.memory_bank['centroids'], cls_score, pseudo_labels) 
            change_ratio = model.odc_classifier.update_memory_bank_samples(idx, features.detach())
            losses['change_ratio'] = change_ratio
            loss_ODC = model.odc_classifier.parse_losses(losses)
            loss += loss_ODC

            scaler.scale(loss).backward()
            scaler.step(optimizer_odc_classifier)
            scheduler_odc_classifier.step()
            scaler.step(optimizer_energy)
            scheduler_energy.step()
            scaler.step(optimizer_flow) 
            scheduler_flow.step(loss)
            scaler.step(optimizer_byte_payload)
            scaler.step(optimizer_byte_header)
            scaler.update()
            total_loss += loss.item()
             
            max_values, predicted_labels = torch.max(cls_score[0], 1)

            features_list.append(features.detach())
            cluster_labels.append(predicted_labels)
            model.odc_classifier.update_centroids()
            model.odc_classifier.deal_with_small_clusters()
            
            _, y_train_pred = torch.max(outputs, dim=1)
            pred_all.extend(y_train_pred.cpu().numpy().tolist())
            target_all.extend(labels_train.cpu().numpy().tolist())
        
        accuracy = accuracy_score(target_all, pred_all)
        precision = precision_score(target_all, pred_all, average='macro')
        recall = recall_score(target_all, pred_all, average='macro')
        f1 = f1_score(target_all, pred_all, average='macro')
        average_loss = total_loss / batch_count

        print(f"Epoch {i + 1} train Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Loss: {average_loss:.4f}")

        pred_all = []
        target_all = []
        data_loader_flow_valid_known.reflesh()
        for data_iter_step, ((samples_header_val, targets_header_val), 
                             (samples_payload_val  , targets_payload_val)) in enumerate(
                        zip(data_loader_byte_header_val_know, 
                            data_loader_byte_payload_val_know)):
            samples_header_val = samples_header_val.to(device, non_blocking=True)
            targets_header_val = targets_header_val.to(device, non_blocking=True)
            
            samples_payload_val = samples_payload_val.to(device, non_blocking=True)
            targets_payload_val = targets_payload_val.to(device, non_blocking=True)
            
            graphs_val, labels_val = data_loader_flow_valid_known.next_valid_batch_know(args.batch_size)  
            if use_gpu :
                graphs_val = graphs_val.to(torch.device(device))
                labels_val = labels_val.to(torch.device(device))
            if mixup_fn is not None:
                samples_header_val, targets_header_val = mixup_fn(samples_header_val, targets_header_val)
                samples_payload_val, targets_payload_val = mixup_fn(samples_payload_val, targets_payload_val)
            with torch.cuda.amp.autocast(enabled=False):
                _, outputs = model(samples_header_val, samples_payload_val, graphs_val, 'energy')
            
            _, y_val_pred = torch.max(outputs, dim=1)
            pred_all.extend(y_val_pred.cpu().numpy().tolist())
            target_all.extend(labels_val.cpu().numpy().tolist())
            
        accuracy = accuracy_score(target_all, pred_all)
        precision = precision_score(target_all, pred_all, average='macro')
        recall = recall_score(target_all, pred_all, average='macro')
        f1 = f1_score(target_all, pred_all, average='macro')
        print()
        print(f"Epoch {i + 1} valid Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 100)

        
        new_labels = model.odc_classifier.memory_bank['labels']
        if new_labels.is_cuda:
            new_labels = new_labels.cpu()
        
        cluster_labels_tensor = torch.cat(cluster_labels, dim=0)
        
        cluster_labels_numpy = cluster_labels_tensor.cpu().numpy()
        true_unknow_labels_tensor = torch.cat(true_unknown_labels_list, dim=0)
        true_unknow_labels_numpy = true_unknow_labels_tensor.cpu().numpy()

        ami_score = adjusted_mutual_info_score(true_unknow_labels_numpy, cluster_labels_numpy)
        print(f'Test ami_score: {ami_score:.2f}')
        
    save_path = 'dataset/saved_models/' + dataset_name + '_FEC_OSL_model_' + \
        str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.pth'
    torch.save(model, save_path)
    
def merge_and_shuffle(features_list, labels_list):
    torch.manual_seed(256)
    np.random.seed(256)
    merged_features = torch.cat(features_list, dim=0)
    merged_labels = torch.cat(labels_list, dim=0)
    
    num_samples = merged_features.shape[0]
    indices = torch.randperm(num_samples)
    
    shuffled_features = merged_features[indices]
    shuffled_labels = merged_labels[indices]
    
    return shuffled_features, shuffled_labels

def merge_and_shuffle_graphs(graphs_list, labels_list):
    torch.manual_seed(256)
    np.random.seed(256)
    merged_graphs = []
    merged_labels = []
    
    for graph_list in graphs_list:
        merged_graphs.append(graph_list)
    for label_list in labels_list:
        merged_labels.append(label_list)
    
    num_samples = len(merged_labels)
    indices = torch.randperm(num_samples)
    
    shuffled_graphs = [merged_graphs[idx] for idx in indices]
    shuffled_labels = torch.tensor([merged_labels[idx] for idx in indices])
    
    return shuffled_graphs, shuffled_labels

def test(model, nb_classes, use_gpu, device, dataset_name, 
        data_loader_byte_header_train_know, data_loader_byte_header_valid_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_valid_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_valid_unknow,
        data_loader_byte_payload_valid_unknow,
        data_loader_flow_valid_unknown,
        ratio):
    for param in model.byte_header_extractor.parameters():
        param.requires_grad = False
    for param in model.byte_payload_extractor.parameters():
        param.requires_grad = False
    for param in model.flow_extractor.parameters():
        param.requires_grad = False
    for param in model.energy_classifier.parameters():
        param.requires_grad = False
    for param in model.odc_classifier.parameters():
        param.requires_grad = False
    
    model.to(device)
    model.byte_header_extractor.to(device)
    model.byte_payload_extractor.to(device)
    model.flow_extractor.to(device)
    model.energy_classifier.to(device)
    model.odc_classifier.to(device)
    
    model.eval()
    model.byte_header_extractor.eval()
    model.byte_payload_extractor.eval()
    model.flow_extractor.eval()
    model.energy_classifier.eval()
    model.odc_classifier.eval()
    
    save_dataloader_to_h5(data_loader_byte_header_train_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_train_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    save_dataloader_to_h5(data_loader_byte_header_valid_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_valid_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    save_dataloader_to_h5(data_loader_byte_header_valid_unknow, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_valid_unknow_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    
    save_dataloader_to_h5(data_loader_byte_payload_train_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_train_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    save_dataloader_to_h5(data_loader_byte_payload_valid_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_valid_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    save_dataloader_to_h5(data_loader_byte_payload_valid_unknow, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_valid_unknow_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    graphs_train_known = []
    labels_train_known = []
    graphs_valid_known = []
    labels_valid_known = []
    graphs_valid_unknown = []
    labels_valid_unknown = []
    for i in data_loader_flow_train_known.known_train_index:
        graphs_train_known.append(data_loader_flow_train_known.graphs[i])
        labels_train_known.append(data_loader_flow_train_known.labelId[i])
    for i in data_loader_flow_valid_known.known_valid_index:
        graphs_valid_known.append(data_loader_flow_valid_known.graphs[i])
        labels_valid_known.append(data_loader_flow_valid_known.labelId[i])
    for i in data_loader_flow_valid_unknown.unknown_valid_index:
        graphs_valid_unknown.append(data_loader_flow_valid_unknown.graphs[i])
        labels_valid_unknown.append(data_loader_flow_valid_unknown.labelId[i])

    graphs_weibull_known = dgl.batch(graphs_train_known)
    labels_weibull_known = torch.tensor(labels_train_known)

    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_header_weibull_know = torch.tensor(h5f['features'])
        label_byte_header_weibull_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_payload_weibull_know = torch.tensor(h5f['features'])
        label_byte_payload_weibull_know = torch.tensor(h5f['labels'])
    
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_valid_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_header_valid_know = torch.tensor(h5f['features'])
        label_byte_header_valid_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_valid_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_payload_valid_know = torch.tensor(h5f['features'])
        label_byte_payload_valid_know = torch.tensor(h5f['labels'])
    
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_valid_unknow_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_header_valid_unknow = torch.tensor(h5f['features'])
        label_byte_header_valid_unknow = torch.tensor(h5f['labels'])
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_valid_unknow_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_payload_valid_unknow = torch.tensor(h5f['features'])
        label_byte_payload_valid_unknow = torch.tensor(h5f['labels'])

    feature_byte_header_valid, label_byte_header_valid = \
        merge_and_shuffle([feature_byte_header_valid_know, feature_byte_header_valid_unknow],
                        [label_byte_header_valid_know, label_byte_header_valid_unknow])
    feature_byte_payload_valid, label_byte_payload_valid = \
        merge_and_shuffle([feature_byte_payload_valid_know, feature_byte_payload_valid_unknow],
                        [label_byte_payload_valid_know, label_byte_payload_valid_unknow])
    
    flow_graphs_valid, flow_labels_valid = merge_and_shuffle_graphs(
        graphs_valid_known + graphs_valid_unknown,
        labels_valid_known +  labels_valid_unknown
    )
    
    T = 10
    threshold = 0.05
    with torch.no_grad():
        _, weibull_logits = model(feature_byte_header_weibull_know.to(device),
            feature_byte_payload_weibull_know.to(device),
            graphs_weibull_known.to(device), 'energy')
        know_energy = energy.calculate_energy(weibull_logits, T).cpu().numpy()
    shape, loc, scale = energy.weibull_min.fit(-know_energy)
    
    labels_valid = flow_labels_valid
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            feature_matrix, logits = model(feature_byte_header_valid.to(device), feature_byte_payload_valid.to(device), dgl.batch(flow_graphs_valid).to(device), 'energy')
            test_energy = energy.calculate_energy(logits).detach().cpu().numpy()
            cdf_values = weibull_min.cdf(-test_energy, shape, loc, scale)
            test_mask = cdf_values < threshold
            test_tensor = torch.tensor(test_mask, dtype=torch.bool)
            
            test_unknown_data = feature_matrix[test_tensor].to(device)
            test_unknown_label = labels_valid[test_tensor].to(device)
            
            _, outputs = model(feature_byte_header_valid_know.to(device), feature_byte_payload_valid_know.to(device), dgl.batch(graphs_valid_known).to(device), 'energy')
            _, y_test_pred = torch.max(outputs, dim=1)
            test_known_label_list = label_byte_header_valid_know.cpu().numpy()
            y_test_pred_list = y_test_pred.cpu().numpy()
            
            accuracy = accuracy_score(test_known_label_list, y_test_pred_list)
            precision = precision_score(test_known_label_list, y_test_pred_list, average='macro')
            recall = recall_score(test_known_label_list, y_test_pred_list, average='macro')
            f1 = f1_score(test_known_label_list, y_test_pred_list, average='macro')
            
            print(f"Known Classfication Test:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 100)
            
            y_test = labels_valid.tolist()
            y_unknown_detected = test_unknown_label.tolist()

            count_dict_test = {}
            for num in y_test:
                if num in count_dict_test:
                    count_dict_test[num] += 1
                else:
                    count_dict_test[num] = 1
            count_dict_unknown = {}
            for num in y_unknown_detected:
                if num in count_dict_unknown:
                    count_dict_unknown[num] += 1
                else:
                    count_dict_unknown[num] = 1
            
            know_list = [i for i in range(int(nb_classes * ratio))]
            unknow_list = [i for i in range(int(nb_classes * ratio), nb_classes)]
            TP = sum(count_dict_unknown.get(cls, 0) for cls in unknow_list)
            FP = sum(count_dict_unknown.get(cls, 0) for cls in know_list)

            FN = sum(count_dict_test.get(cls, 0) for cls in unknow_list) - TP
            TN = sum(count_dict_test.get(cls, 0) for cls in know_list) - FP

            accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print("Unknown Recognition Test:")
            print(f'Accuracy: {accuracy:.4f}')
            print(f'F1 Score: {f1:.4f}')
            y_true = [1] * TP + [0] * TN + [1] * FN + [0] * FP  
            y_scores = [1] * TP + [0] * TN + [0] * FN + [1] * FP  

            auc_score = energy.roc_auc_score(y_true, y_scores)
            print(f'AUC: {auc_score:.4f}')

            cluster_labels = []
            features = model.odc_classifier.extract_feat(test_unknown_data)
            logits = model.odc_classifier.cls_head(features)
            _, cluster_labels = torch.max(logits, dim=1)
            cluster_labels = cluster_labels.cpu().numpy()
            
            true_labels = test_unknown_label
            cluster_labels_numpy = torch.tensor(cluster_labels).cpu().numpy()
            true_labels_numpy = true_labels.clone().detach().cpu().numpy()
            ami_score = adjusted_mutual_info_score(true_labels_numpy, cluster_labels_numpy)
            print(f'Test ami_score: {ami_score:.4f}')
                          
def main():
    nb_classes = 20
    ratio = 0.8
    new_classes = math.ceil((nb_classes - int(nb_classes * ratio)) / 2)

    
    flows_pcap_path = "dataset/pcap/USTC-TFC2016_PCAP"
    if not os.path.exists("dataset/byte_data/USTC-TFC2016_" + 
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio))):
        os.makedirs("dataset/byte_data/USTC-TFC2016_" + 
            str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)))
    output_header_path = "dataset/byte_data/USTC-TFC2016_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) +  "/header"
    output_payaload_path = "dataset/byte_data/USTC-TFC2016_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) +  "/payload"
    
    data_process_header.Grayscale_Image_generator(flows_pcap_path, output_header_path)
    data_process_payload.Grayscale_Image_generator(flows_pcap_path, output_payaload_path)

    output_flow_path = "dataset/flow_data/USTC-TFC2016_JSON_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio))
    if os.path.exists("data/flow_USTC-TFC2016_JSON_" + str(int(nb_classes * ratio)) +  
        "_" + str(nb_classes - int(nb_classes * ratio)) + "/dataset_builder.pkl_" + 
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + ".gzip"):
        print("dataset_builder.pkl.gzip already exists.")
    else:
        pacp_to_json.to_json(flows_pcap_path, output_flow_path, nb_classes, int(nb_classes * ratio))
    
    dataset_name = "USTC-TFC2016"
    dataset_header_path = "dataset/byte_data/USTC-TFC2016_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) +  "/header"
    dataset_payload_path = "dataset/byte_data/USTC-TFC2016_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) +  "/payload"
    dataset_flow_path = 'USTC-TFC2016_JSON_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio))
    
    use_gpu = torch.cuda.is_available()
    device = "cuda:1"
    model = FECOSLModel(nb_classes, use_gpu, device, ratio)

    data_loader_byte_header_train_know,\
    data_loader_byte_header_valid_know,\
    data_loader_byte_header_train_unknow,\
    data_loader_byte_header_valid_unknow = get_byte_header_dataset(dataset_header_path, ratio, new_classes)
    
    data_loader_byte_payload_train_know,\
    data_loader_byte_payload_valid_know,\
    data_loader_byte_payload_train_unknow,\
    data_loader_byte_payload_valid_unknow = get_byte_payload_dataset(dataset_payload_path, ratio, new_classes)
    
    data_loader_flow_train_known,\
    data_loader_flow_valid_known,\
    data_loader_flow_train_unknown,\
    data_loader_flow_valid_unknown = get_flow_dataset(dataset_flow_path, nb_classes, int(nb_classes * ratio))
    
    
    model = train(model, nb_classes, use_gpu, device, dataset_name,
        data_loader_byte_header_train_know, data_loader_byte_header_valid_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_valid_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown,
        ratio
    )
    
    test(model, nb_classes, use_gpu, device, dataset_name,
        data_loader_byte_header_train_know, data_loader_byte_header_valid_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_valid_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_valid_unknow,
        data_loader_byte_payload_valid_unknow,
        data_loader_flow_valid_unknown,
        ratio
    )
    
if __name__ == '__main__':
    main()