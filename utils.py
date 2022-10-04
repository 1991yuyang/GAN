import numpy as np


def imgs_from_tensor(g_output):
    g_output_numpy = g_output.cpu().detach().numpy()
    cv2_imgs = [(127.5 * np.transpose(g_output, axes=[1, 2, 0]) + 127.5).astype(np.uint8) for g_output in g_output_numpy]
    return cv2_imgs