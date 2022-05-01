set -ex
python train.py --dataroot /mnt/now/HADR/data/train --name seg_v1 --model seg --netG unet_256 --mode seg --direction AtoB --dataset_mode image2mask_floodnet --norm batch --num_threads 64 --load_size 512 --crop_size 512  --pool_size 100 --batch_size 32 --n_epochs 10000
