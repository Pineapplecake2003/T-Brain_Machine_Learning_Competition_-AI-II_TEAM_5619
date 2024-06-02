import cv2
import torch
import numpy as np
from PIL import Image

# helper functions
def processbar(now_process, all, total_len = 20, info = ""):
    percent = (now_process / all)
    bar = "■" * int(percent * total_len)
    space = " "*(total_len-len(bar))

    print("|"+bar + space+"|"+info, end='\r' if len(space) > 0 else '\n')

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def record_log(train_loss, vaild_loss, epoch, batch_size, best_acc, learing_rate):
    with open("./log/" + "train_loss_log.txt","w") as f:
        for sig_loss in train_loss:
            f.write(str(sig_loss) + ",")
        f.close()

    with open("./log/" + "vaild_loss_log.txt","w") as f:
        for sig_loss in vaild_loss:
            f.write(str(sig_loss) + ",")
        f.close()

    with open("./log/" + "best_acc_log.txt","w") as f:
        f.write(f"{best_acc}")
        f.close()

    with open("./log/" + "batch_size_log.txt","w") as f:
        f.write(f"{batch_size}")
        f.close()

    with open("./log/" + "learing_rate_log.txt","w") as f:
        for lr in learing_rate:
            f.write(str(lr) + ",")
        f.close()

    with open("./log/" + "epoch_log.txt","w") as f:
        f.write(f"{epoch}")
        f.close()

def convert_from_cv2_to_image(img: np.ndarray):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def get_edge(image):
    img = convert_from_image_to_cv2(image)
    img = cv2.blur(img,(5,5))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, 30, 150)
    return convert_from_cv2_to_image(canny)

def lrfn(num_epoch, optimizer):
    lr_inital = 1e-5  
    max_lr = 4e-4 
    lr_up_epoch = 10
    lr_sustain_epoch = 5  
    lr_exp = .8  
    if num_epoch < lr_up_epoch:  
        lr = (max_lr - lr_inital) / lr_up_epoch * num_epoch + lr_inital
    elif num_epoch < lr_up_epoch + lr_sustain_epoch:  
        lr = max_lr
    else:  
        lr = (max_lr - lr_inital) * lr_exp ** (num_epoch - lr_up_epoch - lr_sustain_epoch) + lr_inital
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer