set -ex
python train.py --dataroot /mnt/now/HADR/data/train --name pix2pix_mask2img_v2 --model pix2pix --netG unet_256 --netD n_layers --n_layers_D 6 --gan_mode lsgan --direction AtoB --dataset_mode mask2image_floodnet --norm batch --num_threads 64 --load_size 512 --crop_size 512  --gpu_ids 0,1 --pool_size 100 --batch_size 35 --n_epochs 2230
