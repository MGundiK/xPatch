import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
# fix_seed = 2022
# fix_seed = 2023
# fix_seed = 2024
# fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# torch.cuda.manual_seed_all(fix_seed)

parser = argparse.ArgumentParser(description='xPatch')

# basic configs
parser.add_argument('--is_training', type=int, required=True, help='status')
parser.add_argument('--train_only', type=bool, default=False)

parser.add_argument('--model_id', type=str, required=True, help='model id')
parser.add_argument('--model', type=str, required=True, help='model name, e.g., xPatch')

# data loader
parser.add_argument('--data', type=str, required=True, help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, e.g., h (hour), t (minute)')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

# patching
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='patch stride')
parser.add_argument('--padding_patch', type=str, default='end', help='None: no padding, end: pad to end')

# decomposition
parser.add_argument('--ma_type', type=str, default='ema', help='decomposition type: ema | dema | gauss | gauss2')
parser.add_argument('--alpha', type=float, default=0.3, help='EMA smoothing coefficient')
parser.add_argument('--beta', type=float, default=0.3, help='DEMA smoothing coefficient')

# ---------------- NEW: Gaussian decomposition parameters ----------------
# Minimal additions so your Model/DECOMP can read them.
parser.add_argument('--gauss_sigma1', type=float, default=None, help='Gaussian sigma for first pass (overrides alpha->sigma mapping)')
parser.add_argument('--gauss_sigma2', type=float, default=None, help='Gaussian sigma for second pass (gauss2)')
parser.add_argument('--gauss_P1', type=int, default=None, help='Period (samples) for first pass; used with gauss_mult1 to compute sigma1')
parser.add_argument('--gauss_mult1', type=float, default=None, help='Multiplier for sigma1 (sigma1 = P1 * mult1)')
parser.add_argument('--gauss_P2', type=int, default=None, help='Period (samples) for second pass (gauss2)')
parser.add_argument('--gauss_mult2', type=float, default=None, help='Multiplier for sigma2 (sigma2 = P2 * mult2)')
parser.add_argument('--gauss_learnable', action='store_true', help='Learnable sigma(s)')
parser.add_argument('--gauss_truncate', type=float, default=4.0, help='Kernel truncation radius in sigmas')
# -----------------------------------------------------------------------

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer initial learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

# hardware & tricks
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')
parser.add_argument('--revin', type=int, default=1, help='RevIN switch')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', help='test flops')

args = parser.parse_args()

# gpu validation
if args.use_gpu and torch.cuda.is_available():
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '').split(',')
        args.gpu = int(args.devices[0])
    else:
        args.devices = None
else:
    args.use_gpu = False

print('Args in experiment:')
print(args)

Exp = Exp_Main
if args.is_training:
    for ii in range(args.itr):
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.des,
            ii
        )

        exp = Exp(args)  # set experiments
        if not args.train_only:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
        else:
            print('>>>>>>>train only : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.des,
        ii
    )
    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    torch.cuda.empty_cache()
