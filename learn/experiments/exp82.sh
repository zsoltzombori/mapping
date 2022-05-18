# parent: exp81
# grid search for d_model
# epochs: 200 -> 100
# num_layers: 1

exp=82
schema="cmt_renamed"
basedir="outdata/cmt_renamed"
epochs=100
batch_size=80
neg_weight=100
num_layers=1
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

d_models=( 128 64 32 16 8 4 )
gpus=( 4 4 4 3 3 3 )

for i in "${!d_models[@]}";
do
    d_model="${d_models[i]}"
    GPU="${gpus[i]}"
    outdir="${outdir_base}/${d_model}"
    mkdir -p $outdir
    CMD="python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER --loss_type ${loss_type} --split ${split} --outdir ${outdir} --monitor_probs $monitor_probs" 
    CMD="nohup $CMD > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
    CMD="CUDA_VISIBLE_DEVICES=$GPU $CMD"
    echo $CMD
    eval $CMD
done
