import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    """Сложить тензоры по кольцу и вернуть часть суммы с индексом rank."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if tensor.ndim != 1 or tensor.device.type != "cpu":
        raise ValueError("Ожидается одномерный CPU-тензор.")
    if tensor.numel() % world_size != 0:
        raise ValueError("Размер тензора должен делиться на число процессов.")
    if tensor.numel() == 0:
        return tensor.clone()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    # Сдвиг на один чанк выбран так, чтобы полная сумма осталась в chunks[rank].
    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        recv_index = (rank - step - 2) % world_size
        received = torch.empty_like(chunks[recv_index])

        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_index], next_rank),
            dist.P2POp(dist.irecv, received, prev_rank),
        ])
        # Обе операции запущены до ожидания; отправляемый буфер пока не меняем.
        for request in requests:
            request.wait()
        chunks[recv_index].add_(received)

    return chunks[rank]


def ring_all_gather(shard):
    """Собрать части по порядку rank в полный тензор на каждом процессе."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if shard.ndim != 1 or shard.device.type != "cpu":
        raise ValueError("Ожидается одномерный CPU-тензор.")
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    if shard.numel() == 0:
        return torch.cat(chunks)

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    for step in range(world_size - 1):
        # Сначала отправляем свою часть, затем часть, полученную на прошлом шаге.
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_index], next_rank),
            dist.P2POp(dist.irecv, chunks[recv_index], prev_rank),
        ])
        for request in requests:
            request.wait()

    return torch.cat(chunks)


def ring_all_reduce(tensor):
    """Вернуть полную сумму на каждом процессе, не изменяя входной тензор."""
    return ring_all_gather(ring_reduce_scatter(tensor))
