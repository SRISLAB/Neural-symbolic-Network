# Model Training
from Dependent_packages.Logger import set_logger
from Dependent_packages.Train_Utils import train_utils
import argparse
import os
from datetime import datetime
import logging


args = None

current_directory = os.getcwd()
dataset_directory = 'CWRU'
data_dir = os.path.join(current_directory, dataset_directory)

def parse_args():

    parser = argparse.ArgumentParser(description='Train')

    # Training arguments
    parser.add_argument('--model_name', type=str, default='TLN', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='CWRU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default=data_dir,
                        help='the directory of the data')
    parser.add_argument('--normalize_type', type=str, choices=['0-1', '-1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./Checkpoint', help='the directory to save the model')
    parser.add_argument("--pre_trained", type=bool, default=True, help='whether to load the pre-trained model')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # Optimizer arguments
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.0005, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')

    # Number of epoch
    parser.add_argument('--max_epoch', type=int, default=400, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    sub_dir = args.model_name + '_' + args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    set_logger(os.path.join(save_dir, 'training.log'))

    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()





