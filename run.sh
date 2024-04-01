# nohup python main.py --name rounD_map02-08_interval100_seq1000_reduced_nfea100 \
# --config_file /home/rzhou/Projects/Diffusion-TS/Config/ika_multi.yaml --gpu 3 --train \
# >> /home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_map02-08_interval100_seq1000_reduced_nfea100.log 2>&1



nohup python main.py --name inD_map19_interval100_seq1000_reduced_nfea40 \
--config_file /home/rzhou/Projects/Diffusion-TS/Config/ika_multi.yaml --gpu 3 --sample 0 --milestone 10 \
>> /home/rzhou/Projects/Diffusion-TS/OUTPUT/inD_map19_interval100_seq1000_reduced_nfea40.log 2>&1