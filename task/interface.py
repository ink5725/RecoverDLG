import torch
from PIL import Image
from module import LogoNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import pickle
import time

def label_to_onehot(target):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), 64, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

if __name__ == "__main__":
    device = "cpu"
    criterion = cross_entropy_for_onehot
    tt = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    tp = transforms.ToPILImage()
    seed = int(time.time()) // 100000
    torch.manual_seed(seed)
    model = LogoNet()
    model.to(device)
    model.apply(weights_init)
    img = Image.open(f'data/logo.png').convert('RGB')
    img_tensor = tt(img)
    gt_label = torch.Tensor([int(0)]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)
    out = model(img_tensor)
    y = criterion(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, model.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    with open(f'model/logo.pkl', 'wb') as f:
        pickle.dump(original_dy_dx, f)
        