import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from communications import all_to_all, all_to_all_single

def benchmark_all_to_all():
    # create inputs
    torch.manual_seed(1024)
    x1 = torch.randn(16, 8096, 32, 512).cuda()
    x2 = x1.clone().detach()

    # get dim-related vals
    num_dim = x1.dim()
    world_size = dist.get_world_size()
    dims = list(range(num_dim))

    # iterate over all combinations of scatter-gather dims
    for scatter_dim in dims:
        for gather_dim in dims:
            if scatter_dim == gather_dim:
                continue
            else:
                # warm up
                for _ in range(5):
                    o1 = all_to_all_single(x1, world_size, None, scatter_dim, gather_dim)
                
                # measure all_to_all_single
                torch.cuda.synchronize()
                start1 = time.time()
                for _ in range(5):
                    o1 = all_to_all_single(x1, world_size, None, scatter_dim, gather_dim)
                torch.cuda.synchronize()
                end1 = time.time()


                # warm up
                for _ in range(5):
                    o2 = all_to_all(x2, world_size, None, scatter_dim, gather_dim)
                
                # measure all_to_all
                torch.cuda.synchronize()
                start2 = time.time()
                for _ in range(5):
                    o2 = all_to_all(x2, world_size, None, scatter_dim, gather_dim)
                torch.cuda.synchronize()
                end2 = time.time()

                # get accuracy
                assert torch.allclose(o1, o2, atol=1e-5, rtol=1e-4), f"{o1}\nvs\n{o2}"

                # print time taken
                print(f"scatter {scatter_dim} and gather {gather_dim}: all to all single = {end1 - start1} s, all to all = {end2 - start2} s ")

def run_dist(rank, world_size, port):
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=f"tcp://localhost:{port}")
    torch.cuda.set_device(rank)
    benchmark_all_to_all()

def main(nprocs=4):
    mp.spawn(run_dist, nprocs=nprocs, args=(nprocs, 29500))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proccess", type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.proccess)
