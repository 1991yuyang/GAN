import numpy as np
import torch as t


def imgs_from_tensor(g_output):
    g_output_numpy = g_output.cpu().detach().numpy()
    cv2_imgs = [(127.5 * np.transpose(g_output, axes=[1, 2, 0]) + 127.5).astype(np.uint8) for g_output in g_output_numpy]
    return cv2_imgs


def calc_grad(model):
    total_grad = []
    for name, parameter in model.named_parameters():
        total_grad.append(t.mean(t.abs(parameter.grad)).item())
    avg_grad = np.mean(total_grad)
    return avg_grad


def clip_weight(discriminator, min, max):
    for name, parameter in discriminator.named_parameters():
        t.clamp_(parameter.data, min, max)
    return discriminator