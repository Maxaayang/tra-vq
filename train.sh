# python3 train.py config/default.yaml > ./logs/test.log
# python3 train.py config/default.yaml > ./logs/vqvae10.log

# rm -rf /home/u21s052015/code/tra-vq/ckpt/or_vae10-A/*
# python train.py config/default.yaml > ./logs/or_vae/or_vae10.log

rm -rf ./generations/or_vae1/*
python3 generate.py config/default.yaml ./ckpt/or_vae1/params/step_143-RC_4.627-KL_0.304-model.pt \
generations/or_vae1/ 10 5 > ./logs/or_gen1.log