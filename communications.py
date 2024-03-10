import torch
import torch.distributed as dist


__all__ = ['all_to_all_single', 'all_to_all']


def all_to_all_single(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    # calculate the partition size
    # the scatter dim will be partitioned into equal parts
    # [val] is reshaped into [partition, partitioned_val]
    # where the value of partition is world size
    inp_shape = list(input_.shape)
    partition_size = inp_shape[scatter_dim] // world_size

    # scatter by the scatter_dim
    scattered_shape = inp_shape[:scatter_dim] + [world_size, partition_size] + inp_shape[scatter_dim + 1 : ]
    scatter_input_ = input_.view(scattered_shape)

    # move the dimension of partiton to dim 0 for all_to_all_single
    # the dimension index of partition is actually equal to scatter dim after reshape
    scatter_input_ = scatter_input_.transpose(0, scatter_dim).contiguous()


    # perform all_to_all
    output = torch.empty_like(scatter_input_)
    dist.all_to_all_single(output, scatter_input_, group=group)

    # permute to get the new shape for gathering
    # there are two steps involved:
    # 1. exchange the dimension index of partition and batch
    output_dims = list(range(output.dim()))
    batch_dim = scatter_dim
    output_dims[0], output_dims[batch_dim] = output_dims[batch_dim], output_dims[0]

    # 2. move the partition dimension to be just before the gather dim
    # note that partition dim is actually the batch dim after exchaing
    # so we pop its index value and put it before gather dim
    val = output_dims.pop(batch_dim)
    output_dims.insert(gather_dim, val)
    output = output.permute(output_dims).contiguous()

    # merge the partiton dim and gather dim
    output_shape = list(output.shape)
    final_output_shape = output_shape[:gather_dim] + [ -1 ] + output_shape[gather_dim+2:]
    output = output.view(final_output_shape).contiguous()
    
    return output


def all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()
