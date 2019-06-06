import torch

"""
Implementation of the refinement network for the IODINE model
"""
class RefineNetLSTM(torch.nn.Module):

	def __init__(self, z_dim, channels_in):
		super(RefineNetLSTM, self).__init__()
		
		self.convnet = ConvNet(channels_in)
		self.fc_in = torch.nn.Sequential(torch.nn.Linear(16384,128),torch.nn.ELU())
		self.lstm = torch.nn.LSTMCell(128+4*z_dim, 128, bias=True)
		self.fc_out = torch.nn.Linear(128,2*z_dim)

	def forward(self, x, h, c):
		x_img, x_vec = x['img'], x['vec']
		N,C,H,W = x_img.shape
		conv_codes = self.convnet(x_img)
		conv_codes = self.fc_in(conv_codes.view(N,-1))
		lstm_input = torch.cat((x_vec,conv_codes),dim=1)
		h,c = self.lstm(lstm_input, (h,c))
		return self.fc_out(h), h, c

class ConvNet(torch.nn.Module):
	
	def __init__(self, channels_in):
		super(ConvNet, self).__init__()
	
		self.model = torch.nn.Sequential(
			torch.nn.Conv2d(channels_in,64,kernel_size=3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.AvgPool2d(4))

	def forward(self, x):
		return self.model(x)
