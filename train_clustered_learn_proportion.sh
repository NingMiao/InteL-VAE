#!/bin/bash
gpus=(0) #GPU sharing and multi-GPU
betas=(1.0)
infos=(1 2 3 4 5) #Repeat experiments 
#mappings=(vanilla clustered)
mappings=(clustered)
multiplier1s=(1)
multiplier2s=(0)
cluster_nums=(2)
least_epoch=20
epochs=(200)
gamma=0.0

latent_dim=50
gpu_stat=-1

for ((i=0;i<${#infos[@]};i++)); do
    for ((j=0;j<${#mappings[@]};j++)); do
        for ((k=0;k<${#cluster_nums[@]};k++)); do
            for ((p=0;p<${#multiplier1s[@]};p++)); do
                    
                    let gpu_stat=gpu_stat+1
                    if [ $gpu_stat -eq ${#gpus[@]} ]; then
                        wait
                        echo wait
                        let gpu_stat=0
                    fi
           
                    if [ ${mappings[$j]} = 'vanilla' ]; then
                        python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=$epoch --least_epoch=$least_epoch --beta=1.0 --gamma=$gamma --latent_dim=$latent_dim --dataset='mnist' --mapping='' --info=${infos[$i]} --test_every=500 --cut_by_labels_dim=1 --multiplier1=2 --multiplier2=3 &
                    else
                        python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=${epochs[$p]} --least_epoch=$least_epoch --beta=1.0 --gamma=$gamma --latent_dim=$latent_dim --dataset='mnist' --mapping='clustered' --info=${infos[$i]} --cluster_num=${cluster_nums[$k]} --test_every=500 --cut_by_labels_dim=1 --multiplier1=${multiplier1s[$p]} --multiplier2=${multiplier2s[$p]} &
                    fi
            done
        done
    done
done
