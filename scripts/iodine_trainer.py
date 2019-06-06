import torchvision.models as models
import torch
import os

from src.iodine import IODINE
from src.networks.refine_net import RefineNetLSTM
from src.networks.sbd import SBD
from src.datasets.datasets import ClevrDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

## Paths for saving models and loading data
save_path = '/home/mkelly2/iodine_clean/IODINE/trained_models/'
datapath = '/home/mkelly2/iodine/data/CLEVR_v1.0/images/'
model_name = 'iodine_clevr_wfeatures'
save_path += model_name + '/'

## Training Parameters
device = 'cuda:0'
batch_size = 32
lr = 3e-4
regularization = 0.
n_epochs = 100
parallel = True
num_workers = 4

## Data Parameters
max_num_samples = 50000
crop_sz = 120 ## Crop initial image down to square image with this dimension
down_sz = 64 ## Rescale cropped image down to this dimension

## Model Hyperparameters
T = 5 ## Number of steps of iterative inference
K = 11 ## Number of slots
z_dim = 64 ## Dimensionality of latent codes
channels_in = 16+16 ## Number of inputs to refinement network (16, + 16 additional if using feature extractor)
out_channels = 4 ## Number of output channels for spatial broadcast decoder (RGB + mask logits channel)
img_dim = (64,64) ## Input image dimension
beta = 5. ## Weighting on nll term in VAE loss
use_feature_extractor = True

## Create training data
train_data = torch.utils.data.DataLoader(
	ClevrDataset(datapath,max_num_samples=max_num_samples,crop_sz=crop_sz,down_sz=down_sz),
	batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

## Create refinement network, decoder, and (optionally) feature extractor
## 		Could speed up training by pre-computing squeezenet outputs since we just use this as a feature extractor
## 		Could also do this as a pre-processing step in dataset class
feature_extractor = models.squeezenet1_1(pretrained=True).features[:5] if use_feature_extractor else None
refine_net = RefineNetLSTM(z_dim,channels_in)
decoder = SBD(z_dim,img_dim,out_channels=out_channels)

## Create IODINE model
v = IODINE(refine_net,decoder,T,K,z_dim,name=model_name,
	feature_extractor=feature_extractor,beta=beta)

## Will use all visible GPUs if parallel=True
if parallel and torch.cuda.device_count() > 1:
  print('Using {} GPUs'.format(torch.cuda.device_count()))
  v = torch.nn.DataParallel(v)
  v_module = v.module
else:
  parallel = False
  v_module = v

## Set up optimizer and data logger
optimizer = torch.optim.Adam(v.parameters(),lr=lr,weight_decay=regularization)
writer = SummaryWriter(save_path+'logs/')


def train(model,dataloader,n_epochs=10,device='cpu'):
	v = model.to(device)
	mbatch_cnt = 0
	
	for epoch in range(n_epochs):
		
		print('On epoch {}'.format(epoch))
		for i,mbatch in enumerate(dataloader):
	
			mbatch = mbatch.to(device)		
			N,C,H,W = mbatch.shape

			## Forward pass
			loss, nll, div, mu_x, masks = v.forward(mbatch)	
			
			## Process Outputs
			if parallel:
				nll = nll.mean()
				div = div.mean()
				loss = loss.mean()
			assert not torch.isnan(loss).item(), 'Nan loss: loss / div / nll: {}/{}/{}'.format(loss,div,nll)
			output_means = (mu_x*masks).sum(dim=1)
			mse = torch.nn.functional.mse_loss(output_means, x)

			## Backwards Pass
			optimizer.zero_grad()
			loss.backward(retain_graph=False)
			torch.nn.utils.clip_grad_norm_(v.parameters(), 5.0, norm_type=2)
			v_module.check_grad()

			## Update model
			assert not v_module.hasnan(), 'Model has nan pre-opt step'
			optimizer.step()
			assert not v_module.hasnan(), 'Model has nan post-opt step'

			## Print and log outputs
			if i%10==0:
				print('\nOn mbatch {}:'.format(mbatch_cnt))
				print('model.b = {}'.format(v_module.b))
				print('Curr loss: {}'.format(loss.item()))
				print('Curr final nll: {}'.format(nll.item()))
				print('Curr final div: {}'.format(div.item()))
				print('Curr mse: {}'.format(mse.item()))
			writer.add_scalar('loss',loss.item(),mbatch_cnt)
			writer.add_scalar('final nll',nll.item(),mbatch_cnt)
			writer.add_scalar('final div',div.item(),mbatch_cnt)
			writer.add_scalar('final mse',mse.item(),mbatch_cnt)
			mbatch_cnt += 1
			
			## Save model at half-epoch increments
			if i==len(dataloader)//2:
				v_module.save(save_path,epoch=epoch+0.5)

		## Save model at half-epoch increments
		if epoch%1==0:
			v_module.save(save_path,epoch=epoch)

## Run training function
train(v,train_data,n_epochs=n_epochs,device=device)
