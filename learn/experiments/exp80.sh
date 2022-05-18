# parent: exp79
# grid search for random seed
# neg_weight: 3 -> 100

exp=80
schema="cmt_renamed"
basedir="outdata/cmt_renamed"
epochs=300
batch_size=40
neg_weight=100
num_layers=4
d_model=128
lr=0.001
lr_decay_steps=30
optimizer=adamax
beta1=0.3
beta2=0.9
CHAR_TOKENIZER=0
loss_type="lprp"
split="0.7,0.15,0.15"
monitor_probs=0
GPU=6

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="${basedir}/${schema}"
outdir_base="out/exp${exp}"

seeds=( 1 2 3 4 5 )
gpus=( 6 5 4 3 2 )

for i in "${!seeds[@]}";
do
    SEED="${seeds[i]}"
    GPU="${gpus[i]}"
    outdir="${outdir_base}/${SEED}"
    mkdir -p $outdir
    CMD="python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER --loss_type ${loss_type} --split ${split} --outdir ${outdir} --monitor_probs $monitor_probs --seed $SEED" 
    CMD="nohup $CMD > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
    CMD="CUDA_VISIBLE_DEVICES=$GPU $CMD"
    echo $CMD
    eval $CMD
done
