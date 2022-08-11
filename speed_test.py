import torch
import time
from model.network import *


def report_speed(model, data, times=100, warm_up_times=30):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model.eval()
    with torch.no_grad():
        for _ in range(warm_up_times):
            pred = model(data)

    start = time.time()
    with torch.no_grad():
        for _ in range(times):
            pred = model(data)
    time_cost = (time.time() - start) / times
    return time_cost


if __name__ == "__main__":
    model = DualResNet_imagenet(pretrain_path='./pretrained_models/DDRNet23s_imagenet.pth', pretrained=True,
                                num_classes=19)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    dummy_input = torch.rand(1, 3, 2048, 1024).to(device)

    time_cost = report_speed(model, dummy_input)

    print('FPS: {:.2f}'.format(1 / time_cost))
