import torch
import argparse
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/srim_RRDB_14_n02279972_461_x8_0.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/srim_RRDB_14_n02279972_461_x8_25000.png')
parser.add_argument('-t','--tar', type=str, default='./imgs/n02279972_461.JPEG')
parser.add_argument('-p','--scale', type=int, default=0.5)


opt = parser.parse_args()

# Load images
img0 = util.im2tensor(util.load_image(opt.path0)) # RGB image from [-1,1]
img1 = util.im2tensor(util.load_image(opt.path1))
target = util.im2tensor(util.load_image(opt.tar))

p = opt.scale
dist0 = torch.norm(img0 - target, p=p)
dist1 = torch.norm(img1 - target, p=p)

print("dist 0: %f" % dist0)
print("dist 1: %f" % dist1)
