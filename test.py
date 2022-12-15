
import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse

from tester import Tester



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--gpu", dest="gpu",required=True)
    arg_parser.add_argument("--save_dir", dest="save_dir", required=True)
    arg_parser.add_argument("--batchsize", dest="batchsize", default = 2048)
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="srncar.json")
    arg_parser.add_argument("--num_instances_per_obj", dest="num_instances_per_obj", default=2)

    args = arg_parser.parse_args()
    save_dir = args.save_dir
    gpu = int(args.gpu)
    B = int(args.batchsize)
    num_instances_per_obj = int(args.num_instances_per_obj)

    tester = Tester(save_dir, gpu, jsonfile=args.jsonfile, batch_size=B)
    tester.testing_epoch(num_instances_per_obj)

