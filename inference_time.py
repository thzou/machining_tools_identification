import torch 
import numpy as np
import importlib
import shutil
import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=16, type=int, choices=[16],  help='training on 16 Tools Dataset')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()

def current_milli_time():
    return round(time.time() * 1000)

def main(args):

    args = parse_args()
    num_class= 16 # 16 Tools

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    model = model.get_model(num_class, normal_channel=args.use_normals)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()



    if args.use_cpu == True:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda")
    model.to(device)
    
    # input = np.load("data/16_Tools_Large_Dataset/tool00/tool00_0028.npy")
    # input = np.load("data/16_Tools_Small_Dataset/tool00/tool00_0028.npy")
    # input = torch.from_numpy(input).float().to(device)
    # input = input.view(1, 3, 1024)

    input = torch.randn(1, 3, 1024, dtype=torch.float).to(device)

    repetitions = 10000 # how many times to run single inference
    timings=np.zeros((repetitions,1))
    if args.use_cpu == True:
        
        with torch.no_grad():

            for rep in range(repetitions):
                t0 = current_milli_time()
                _ = model(input)
                t1 = current_milli_time()
                curr_time = t1 - t0
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(mean_syn, std_syn)

    else:
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # GPU-WARM-UP
        for _ in range(10):
            _ = model(input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(mean_syn, std_syn)

if __name__ == '__main__':
    args = parse_args()
    main(args)
