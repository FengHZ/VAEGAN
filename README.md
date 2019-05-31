# VAEGAN
Here are some code for project combine VAE and GAN

Usage:

1. Put the celeba dataset into the 
   ```
   path.join(basepath,"celeba")
   ```
   root, and in the folder `celeba` you can unzip the raw data `img_align_celeba.zip`

2. Usage
   ```
   python main_celeba.py -t 1 -bp `basepath` -dp --gpu "0,1"
   ```
