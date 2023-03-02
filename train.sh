# source ~/.zshrc
# conda activate col

# python3 train.py config/default.yaml > ./logs/test.log
# rm -rf ./ckpt/vqvae400/*
# python3 train.py config/default.yaml > ./logs/soft/soft100.log

# python3 generate.py config/default.yaml ./ckpt/juke50/params/step_9850-RC_1.760-KL_0.000-model.pt \
# generations/juke50/ 10 5 > ./logs/gen50.log
# python3 train.py config/default.yaml > ./logs/vqvae10.log

# python train.py config/default.yaml > ./logs/vqvae/new_vqvae50.log


# rm -rf ./generations/123vqvae10/*
# python3 generate.py config/default.yaml ./ckpt/vqvae10/params/step_1170-RC_2.458-VQ_0.073-model.pt \
# generations/123vqvae10/ 10 5 > ./logs/gen/123gen_vq10.log

python3 generate-ori.py config/default.yaml ./ckpt/vqvae10/params/step_590-RC_2.721-VQ_0.003-model.pt generations/soft/ 10 5 > ./logs/gen/soft100.log