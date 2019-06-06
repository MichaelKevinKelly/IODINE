import torch
import math

"""
Implementation of  Iterative Object Decomposition Inference Network (IODINE)
from "Multi-Object Representation Learning with Iterative Variational Inference" by Greff et. al. 2019
Link: https://arxiv.org/pdf/1903.00450.pdf
"""
class IODINE(torch.nn.Module):

	def __init__(self,
		refine_net,
		decoder,
		T,
		K,
		z_dim,
		name='iodine',
		beta=1.,
		feature_extractor=None):
		super(IODINE, self).__init__()
		
		self.lmbda0 = torch.nn.Parameter(torch.rand(1,2*z_dim)-0.5,requires_grad=True)
		self.decoder = decoder
		self.refine_net = refine_net
		self.layer_norms = torch.nn.ModuleList([
			torch.nn.LayerNorm((1,64,64),elementwise_affine=False),
			torch.nn.LayerNorm((3,64,64),elementwise_affine=False),
			torch.nn.LayerNorm((1,64,64),elementwise_affine=False),
			torch.nn.LayerNorm((2*z_dim,),elementwise_affine=False),
			torch.nn.LayerNorm((1,64,64),elementwise_affine=False)])
		
		self.use_feature_extractor = feature_extractor is not None
		if self.use_feature_extractor:
			self.feature_extractor = torch.nn.Sequential(
				feature_extractor,
				torch.nn.Conv2d(128,64,3,stride=1,padding=1),
				torch.nn.ELU(),
				torch.nn.Conv2d(64,32,3,stride=1,padding=1),
				torch.nn.ELU(),
				torch.nn.Conv2d(32,16,3,stride=1,padding=1),
				torch.nn.ELU())
			for param in self.feature_extractor[0]:
				param.requires_grad = False
		
		self.name = name
		self.register_buffer('T', torch.tensor(T))
		self.register_buffer('K', torch.tensor(K))
		self.register_buffer('z_dim', torch.tensor(z_dim))
		self.register_buffer('var_x', torch.tensor(0.3))
		self.register_buffer('h0',torch.zeros((1,128)))
		self.register_buffer('base_loss',torch.zeros(1,1))
		self.register_buffer('b', torch.tensor(beta)) ## Weight on NLL component of loss
		self._create_meshgrid()
		self._setup_debug()

	"""
	Forward pass through IODINE model.
	Consists of T steps of iterative inference performed on the parameters of the latent
	variables for each of the K slots / components.
	"""
	def forward(self, x):
				
		N,C,H,W = x.shape
		K, T, z_dim = self.K, self.T, self.z_dim
		assert not torch.isnan(self.lmbda0).any().item(), 'lmbda0 has nan'
		
		## Initialize parameters for latents' distribution
		lmbda = self.lmbda0.expand((N*K,)+self.lmbda0.shape[1:])
		total_loss, losses = torch.zeros_like(self.base_loss.expand((N,1))), []
		
		## Initialize LSTMCell hidden states
		h = self.h0.expand((N*K,)+self.h0.shape[1:]).clone().detach() ##TODO
		c = torch.zeros_like(h)
		assert h.max().item()==0. and h.min().item()==0.

		for it in range(T):
			
			## Sample latent code
			mu_z, logvar_z = lmbda.chunk(2,dim=1)
			mu_z, logvar_z = mu_z.contiguous(), logvar_z.contiguous()
			z = self._sample(mu_z,logvar_z) ## (N*K,z_dim)
			
			## Get means and masks 
			dec_out = self.decoder(z) ## (N*K,C+1,H,W)
			mu_x, mask_logits = dec_out[:,:C,:,:], dec_out[:,C,:,:] ## (N*K,C,H,W), (N*K,H,W)
			mask_logits = mask_logits.view((N,K,)+mask_logits.shape[1:]) ## (N,K,H,W)
			mu_x = mu_x.view((N,K,)+mu_x.shape[1:]) ##(N,K,C,H,W)

			## Process masks
			masks = torch.nn.functional.softmax(mask_logits,dim=1).unsqueeze(dim=2) ##(N,K,1,H,W)
			mask_logits = mask_logits.unsqueeze(dim=2) ##(N,K,1,H,W)

			## Calculate loss: reconstruction (nll) & KL divergence
			_x = x.unsqueeze(dim=1).expand((N,K,)+x.shape[1:]) ## (N,K,C,H,W)
			deviation = -1.*(mu_x - _x)**2
			ll_pxl_channels = ((masks*(deviation/(2.*self.var_x)).exp()).sum(dim=1,keepdim=True)).log()
			assert ll_pxl_channels.min().item()>-math.inf
			ll_pxl = ll_pxl_channels.sum(dim=2,keepdim=True) ## (N,1,1,H,W)
			ll_pxl_flat = ll_pxl.view(N,-1)
			
			nll = -1.*(ll_pxl_flat.sum(dim=-1).mean())
			div = self._get_div(mu_z,logvar_z,N,K)
			loss = self.b * nll + div
			
			## Accumulate loss
			scaled_loss = ((float(it)+1)/float(T)) * loss
			losses.append(scaled_loss)
			total_loss += scaled_loss
			
			assert not torch.isnan(loss).any().item(), 'Loss at t={} is nan. (nll,div): ({},{})'.format(nll,div)
			if it==T-1: continue

			## Refine lambda
			refine_inp = self.get_refine_inputs(_x,mu_x,masks,mask_logits,ll_pxl,lmbda,loss,deviation)

			## Potentially add additional features from pretrained model (scaled down to appropriate size)
			if self.use_feature_extractor:
				x_resized = torch.nn.functional.interpolate(x,257) ## Upscale to desired input size for squeezenet
				additional_features = self.feature_extractor(x_resized)
				additional_features = additional_features.unsqueeze(dim=1)
				additional_features = additional_features.expand((N,K,16,64,64)).contiguous()
				
				print(additional_features.shape)
				additional_features = additional_features.view((N*K,16,64,64))
				print(additional_features.shape)
				print(refine_inp['img'].shape)
				refine_inp['img'] = torch.cat((refine_inp['img'],additional_features),dim=1)
				print(refine_inp['img'].shape)
				assert False

			delta, h, c = self.refine_net(refine_inp, h, c)
			assert not torch.isnan(lmbda).any().item(), 'Lmbda at t={} has nan: {}'.format(it,lmbda)
			assert not torch.isnan(delta).any().item(), 'Delta at t={} has nan: {}'.format(it,delta)
			lmbda = lmbda + delta
			assert not torch.isnan(lmbda).any().item(), 'Lmbda at t={} has nan: {}'.format(it,lmbda)

		return total_loss, nll, div, mu_x, masks

	"""
	Generate inputs to refinement network
	"""
	def get_refine_inputs(self,_x,mu_x,masks,mask_logits,ll_pxl,lmbda,loss,deviation):
		N,K,C,H,W = mu_x.shape
		
		## Calculate additional non-gradient inputs
		ll_pxl = ll_pxl.expand((N,K,) + ll_pxl.shape[2:]) ## (N,K,1,H,W)
		p_mask_individual = (deviation/(2.*self.var_x)).exp().prod(dim=2,keepdim=True) ## (N,K,1,H,W)
		p_masks = torch.nn.functional.softmax(p_mask_individual, dim=1) ## (N,K,1,H,W)
		
		## Calculate gradient inputs
		dmu_x = torch.autograd.grad(loss,mu_x,retain_graph=True,only_inputs=True)[0] ## (N,K,C,H,W)
		dmasks = torch.autograd.grad(loss,masks,retain_graph=True,only_inputs=True)[0] ## (N,K,1,H,W)
		dlmbda = torch.autograd.grad(loss,lmbda,retain_graph=True,only_inputs=True)[0] ## (N*K,2*z_dim)

		## Apply layer norm
		ll_pxl_stable = self.layer_norms[0](ll_pxl).detach()
		dmu_x_stable = self.layer_norms[1](dmu_x).detach()
		dmasks_stable = self.layer_norms[2](dmasks).detach()
		dlmbda_stable = self.layer_norms[3](dlmbda).detach()
		
		## Generate coordinate channels
		x_mesh = self.x_grid.expand(N,K,-1,-1,-1).contiguous()
		y_mesh = self.y_grid.expand(N,K,-1,-1,-1).contiguous()

		## Concatenate into vec and mat inputs
		img_args = (_x,mu_x,masks,mask_logits,dmu_x_stable,dmasks_stable,
			p_masks,ll_pxl_stable,x_mesh,y_mesh)
		vec_args = (lmbda, dlmbda_stable)
		
		## Check inputs for nans
		# for i in range(len(img_args)):
		# 	assert not torch.isnan(img_args[i]).any().item(), 'Img arg {} has nan: {}'.format(i,img_args[i])
		# for i in range(len(vec_args)):
		# 	assert not torch.isnan(vec_args[i]).any().item(), 'Vec arg {} has nan: {}'.format(i,vec_args[i])

		img_inp = torch.cat(img_args,dim=2)
		vec_inp = torch.cat(vec_args,dim=1)

		## Reshape
		img_inp = img_inp.view((N*K,)+img_inp.shape[2:])

		return {'img':img_inp, 'vec':vec_inp}

	"""
	Computes the KL-divergence between an isotropic Gaussian distribution over latents
	parameterized by mu_z and logvar_z and the standard normal
	"""
	def _get_div(self,mu_z,logvar_z,N,K):
		kl = ( -0.5*((1.+logvar_z-logvar_z.exp()-mu_z.pow(2)).sum(dim=1)) ).view((N,K))
		return (kl.sum(dim=1)).mean()

	"""
	Implements the reparameterization trick
	Samples from standard normal and then scales and shifts by var and mu
	"""
	def _sample(self,mu,logvar):
		std = torch.exp(0.5*logvar)
		return mu + torch.randn_like(std)*std

	"""
	Generates coordinate channels inputs for refinemet network
	"""
	def _create_meshgrid(self):
		H,W = (64,64)
		x_range = torch.linspace(-1.,1.,W)
		y_range = torch.linspace(-1.,1.,H)
		x_grid, y_grid = torch.meshgrid([x_range,y_range])
		self.register_buffer('x_grid', x_grid.view((1, 1, 1) + x_grid.shape))
		self.register_buffer('y_grid', y_grid.view((1, 1, 1) + y_grid.shape))

	"""
	Enable post mortem debugging
	"""
	def _setup_debug(self):
		import sys
		old_hook = sys.excepthook

		def new_hook(typ, value, tb):
			old_hook(typ, value, tb)
			if typ != KeyboardInterrupt:
				import ipdb
				ipdb.post_mortem(tb)

		sys.excepthook = new_hook

	"""
	Save the current IODINE model
	"""
	def save(self,save_path,epoch=None):
		print('Saving model at epoch {}'.format(epoch))
		suffix = self.name if epoch is None else self.name+'_epoch_{}.th'.format(epoch)
		model_save_path = save_path + suffix
		torch.save(self.state_dict(),model_save_path)

	"""
	Loads weights for the IODINE model
	"""
	def load(self,load_path,map_location):
		params = torch.load(load_path,map_location=map_location)
		self.load_state_dict(params)

	"""
	Checks if any of the model's weights are NaN
	"""
	def has_nan(self):
		for name,param in self.named_parameters():
			if torch.isnan(param).any().item():
				print(param)
				assert False, '{} has nan'.format(name)

	"""
	Checks if any of the model's weight's gradients are NaNs
	"""
	def grad_has_nan(self):
		for name,param in self.named_parameters():
			if torch.isnan(param.grad).any().item():
				print(param)
				print('---------')
				print(param.grad)
				assert False, '{}.grad has nan'.format(name)
