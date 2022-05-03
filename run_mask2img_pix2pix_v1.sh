set -ex
python train.py --dataroot /mnt/now/HADR/data/train --name pix2pix_mask2img_v1 --model pix2pix --netG unet_256 --netD n_layers --n_layers_D 6 --gan_mode lsgan --direction AtoB --dataset_mode mask2image_floodnet --norm batch --num_threads 64 --load_size 512 --crop_size 512  --mode v1 --pool_size 100 --lr 0.00001 --lambda_ 1.0 --eta_ 100.0 --batch_size 32 --n_epochs 10000
