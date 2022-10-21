# python3 train.py config/default.yaml > ./logs/test.log
# python3 train.py config/default.yaml > ./logs/vqvae10.log

rm -rf /home/u21s052015/code/tra-vq/ckpt/or_vae10/*
python train.py config/default.yaml > ./logs/vae/or_vae10.log

# python3 generate.py config/default.yaml ./ckpt/juke50/params/step_9850-RC_1.760-KL_0.000-model.pt \
# generations/juke50/ 10 5 > ./logs/gen50.log