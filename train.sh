# source ~/.zshrc
# conda activate col

# python3 train.py config/default.yaml > ./logs/test.log
rm -rf ./ckpt/vqvae50/*
python3 train.py config/default.yaml > ./logs/vqvae/vqvae50.log

# python3 generate.py config/default.yaml ./ckpt/juke50/params/step_9850-RC_1.760-KL_0.000-model.pt \
# generations/juke50/ 10 5 > ./logs/gen50.log
# python3 train.py config/default.yaml > ./logs/vqvae10.log

# python train.py config/default.yaml > ./logs/vqvae/new_vqvae50.log


# rm -rf ./generations/vqvae400/*
# python3 generate.py config/default.yaml ./ckpt/vqvae400/params/step_57200-RC_1.266-VQ_0.083-model.pt \
# generations/vqvae400/ 10 5 > ./logs/gen/gen_vq4000.log
