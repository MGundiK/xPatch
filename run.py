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

# decomposition (UPDATED)
parser.add_argument(
    '--ma_type',
    type=str,
    default='ema',
    help='decomposition type: reg | ema | dema | gauss | gauss_adaptive | doghybrid | lp_learnable | tcn_trend'
)
parser.add_argument('--alpha', type=float, default=0.3, help='EMA smoothing coefficient')
parser.add_argument('--beta', type=float, default=0.3, help='DEMA smoothing coefficient')

# -------- Gaussian (single-pass) --------
parser.add_argument('--gauss_sigma1', type=float, default=None, help='Gaussian sigma (overrides alpha->sigma mapping)')
parser.add_argument('--gauss_P1', type=int, default=None, help='Period (samples); used with gauss_mult1 to compute sigma')
parser.add_argument('--gauss_mult1', type=float, default=None, help='Multiplier for sigma (sigma = P1 * mult1)')
parser.add_argument('--gauss_learnable', action='store_true', help='Make Gaussian sigma learnable')
parser.add_argument('--gauss_truncate', type=float, default=4.0, help='Kernel truncation radius in sigmas')

# -------- Adaptive Gaussian (Causal) --------
parser.add_argument('--adaptive_sigmas', type=lambda s: [float(x) for x in s.split(',')],
                    default=[2.5,4.0,6.0,9.0,14.0],
                    help='List of sigmas for causal adaptive Gaussian mixture')
parser.add_argument('--adaptive_truncate', type=float, default=4.0,
                    help='Truncation for adaptive Gaussian kernels')
parser.add_argument('--adaptive_cond_hidden', type=int, default=32,
                    help='Hidden dim for conditioner MLP')
parser.add_argument('--adaptive_stat_window', type=int, default=16,
                    help='Causal window length for local mean/var')
parser.add_argument('--adaptive_add_x_feature', action='store_true',
                    help='Include raw x as a conditioner feature')



# -------- Hybrid EMA + DoG --------
parser.add_argument('--dog_sigma1', type=float, default=4.2, help='DoG small sigma')
parser.add_argument('--dog_sigma2', type=float, default=96.0, help='DoG large sigma')
parser.add_argument('--dog_truncate', type=float, default=4.0, help='Truncation for DoG Gaussians')

# -------- Learnable LP --------
parser.add_argument('--lp_kernel_size', type=int, default=21, help='Kernel size for learnable LP')
parser.add_argument('--lp_mode', type=str, default='centered', help='LP mode: centered | causal')
parser.add_argument('--lp_ema_alpha', type=float, default=0.3, help='EMA-like initialization for LP kernel')

# -------- TCN Smoother --------
parser.add_argument('--tcn_hidden_mult', type=float, default=1.0, help='Hidden width multiplier for TCN smoother')
parser.add_argument('--tcn_n_blocks', type=int, default=2, help='Number of TCN blocks')
parser.add_argument('--tcn_kernel', type=int, default=7, help='Kernel size per depthwise conv in TCN')
parser.add_argument('--tcn_beta', type=float, default=0.3, help='Residual smoothing strength in TCN')
parser.add_argument('--tcn_final_avg', type=int, default=0, help='Final average window; 0 disables')

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

# Parse adaptive_sigmas if provided as CSV
def _parse_float_list_csv(s):
    if s is None:
        return None
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return None
        return [float(x) for x in s.split(',')]
    return s

args.adaptive_sigmas = _parse_float_list_csv(args.adaptive_sigmas)

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
