set -ex
python test.py --dataroot /mnt/now/HADR/data/test --name pix2pix_mask2img_v2 --model pix2pix --netG unet_256 --netD n_layers --n_layers_D 6 --direction AtoB --dataset_mode mask2image_floodnet_inference --norm batch --num_threads 64 --load_size 512 --crop_size 512  --gpu_ids 0
