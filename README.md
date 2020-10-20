GreedyFool
============================
Implemention of GreedyFool: Distortion-Aware Sparse Adversarial Attack (NIPS2020)

* Supports PyTorch version >= 1.0.0.  



Setup
-----

* Install ``python`` -- This repo is tested with ``3.6``
* Install PyTorch version >= 1.0.0, torchvision >= 0.2.1.  


Adversarial Attack
------------------
They can be run via

::

  python -m pointnet2.train.train_cls

  python -m pointnet2.train.train_sem_seg
For white-box attack, run 

::

  bash nips_gd.sh 
  
  --dataroot input image data with png format
  
  --name name of the current test
  
  --phase phase of the current test
  
  --max_epsilon max of the adversarial perturbation [0,255]
  --iter max iteration number for attack
  




