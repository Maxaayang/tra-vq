# python3 train.py config/default.yaml > ./logs/test.log
# python3 train.py config/default.yaml > ./logs/vqvae10.log

rm -rf /home/u21s052015/code/tra-vq/ckpt/new_vae30_kl1430/*
python train.py config/default.yaml > ./logs/new_vae/new_vae30_kl573__klcycle3000.log

# python3 generate.py config/default.yaml ./ckpt/new_vae30_kl573__klcycle3000/params/step_2145-RC_0.061-KL_0.000-model.pt \
# generations/new_vae30_kl573__klcycle3000/ 10 5 > ./logs/new_vae30_kl573__klcycle3000.log
