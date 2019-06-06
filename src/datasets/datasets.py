from torch.utils.data import Dataset
from skimage.transform import rescale, rotate
from skimage import io
import numpy as np
import torch

"""
Pytorch dataset class for loading pre-generated images from the CLEVR dataset
"""
class ClevrDataset(Dataset):

    def __init__(self,datapath,data_type='train',max_num_samples=60000,crop_sz=256,down_sz=64):
        suffix = data_type
        self.datapath = datapath + suffix + '/CLEVR_' + suffix + '_'
        self.max_num_samples = max_num_samples
        self.crop_sz = crop_sz
        self.down_scale = down_sz/crop_sz

    def __len__(self):
        return self.max_num_samples

    def __getitem__(self,idx):
        imgname = self.datapath + str(idx).zfill(7)
        imgpath = imgname + '.png'
        scaled_img = self.rescale_img(io.imread(imgpath))
        img = torch.tensor(scaled_img,dtype=torch.float32).permute((2,0,1))
        return img

    def rescale_img(self,img):
        H,W,C = img.shape
        dH = abs(H-self.crop_sz)//2
        dW = abs(W-self.crop_sz)//2
        crop = img[dH:-dH,dW:-dW,:3]
        down = rescale(crop, self.down_scale, order=3, mode='reflect', multichannel=True)
        return down

def main():
    import matplotlib.pyplot as plt
    data_path_clevr = '/Users/mike/Desktop/Academics/Research/LifelongLearning/PDDL/blockstacking-clean/blockstacking/data/CLEVR_v1.0/images/'
    d = ClevrDataset(data_path_clevr,crop_sz=210,down_sz=64,max_num_samples=40)
    data = torch.utils.data.DataLoader(d,batch_size=2, shuffle=True, num_workers=1)
    
    for i,batch in enumerate(data):
        if i>10: break
        print('On batch {}'.format(i))
        x = batch[0]    
        plt.imshow(x.permute((1,2,0)).detach().numpy())
        plt.show()
        

if __name__=='__main__':
    main()