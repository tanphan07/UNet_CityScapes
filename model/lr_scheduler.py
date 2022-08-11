import numpy as np


def adjust_optim(optimizer, current_iter, total_iters, factor, init_lr, warmup_iters=3000, warmup_factor=0.1):
    rate = np.power(1.0 - current_iter / float(total_iters + 1), factor)
    if current_iter < warmup_iters:
        alpha = warmup_factor + (1 - warmup_factor) * (current_iter / warmup_iters)
        rate *= alpha
        
    optimizer.param_groups[0]['lr'] = rate * init_lr
    return optimizer.param_groups[0]['lr']
