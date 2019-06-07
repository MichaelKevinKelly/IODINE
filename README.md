This repo contains an implementation of the  Iterative Object Decomposition Inference Network (IODINE).  
Paper: https://arxiv.org/pdf/1903.00450.pdf

Dependencies:  
python 3.7  
pytorch 1.1  
numpy 1.16.4  
scikit-image 0.15.0  
tensorboardX 1.7  
PIL 6.0.0  
ipdb 0.12  

To run, adjust parameters in scripts/iodine_trainer.py as desired and then run "scripts/iodine_trainer.py" from the repo's root directory. 

In particular, savepath and datapath will need to be updated. In its current instantation, the dataset class expects that the data directory specified by datapath will include a "train" subdirectory containing consecutively numbered training images named "CLEVR_train_*.png".
