GreedyFool
============================
Implemention of GreedyFool: Distortion-Aware Sparse Adversarial Attack (NIPS2020)


Setup
-----

* Install ``python`` -- This repo is tested with ``3.6``

* Install ``PyTorch version >= 1.0.0, torchvision >= 0.2.1``

* Download Inception-v3 pretrained model from ``https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth`` and move it to ``./pretrain/``
 

Adversarial Attack
------------------
run on ImageNet dataset
	
	bash nips_gd.sh        for white-box attack
	bash nips_black_gd.sh  for black-box attack
	
paramaters 

	--dataroot       input image data with png format
	--name           name of the current test
	--phase          phase of the current test
	--max_epsilon    max of the adversarial perturbation [0,255]
	--iter           max iteration number for attack
	--confidence     confidence factor for control the black-box strength (do not work for the white-box setting)
  

