# flake8: noqa E402
import sys
sys.path.append('../../../tests/third_party/flow_net')
from PIL import Image
from torchvision.transforms.functional import to_tensor
from imaginaire.third_party.flow_net.flow_net import FlowNet
import wget
import os

im1_fn = 'frame_0010.png'
im2_fn = 'frame_0011.png'
im1_url = 'https://github.com/NVlabs/PWC-Net/raw/master/PyTorch/data/' + im1_fn
im2_url = 'https://github.com/NVlabs/PWC-Net/raw/master/PyTorch/data/' + im2_fn
if not os.path.exists(im1_fn):
    wget.download(im1_url, out=im1_fn)
if not os.path.exists(im2_fn):
    wget.download(im2_url, out=im2_fn)
img1 = to_tensor(Image.open(im1_fn).convert('RGB')).unsqueeze(0).cuda()
img2 = to_tensor(Image.open(im2_fn).convert('RGB')).unsqueeze(0).cuda()
# Image range between -1 and 1
img1 = 2 * img1 - 1
img2 = 2 * img2 - 1
flow_net = FlowNet()
flow = flow_net.cuda().eval()
flo, conf = flow(img1, img2)
print(flo.size())
print(conf.size())
