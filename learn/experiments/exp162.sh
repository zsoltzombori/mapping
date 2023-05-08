# parent: exp88, exp91
# epochs: 20
# logit_decay

exp=162
schema="cmt_renamed"
basedir="outdata/cmt_renamed"
epochs=20
batch_size=20 #40
neg_weight=3
num_layers=4
d_model=128
lr=0.0001
lr_decay_steps=30
optimizer=adamax
beta1=0.3
beta2=0.9
CHAR_TOKENIZER=0
loss_type="lprp"
split="0.7,0.15,0.15"
monitor_probs=0
GPU=1

logitdecays=( 0.0 0.1 0.01 0.001 )
gpus=( 1 2 3 5 )

outdir_base="out/exp${exp}"
checkpoint_dir_base="checkpoints/exp${exp}"

for i in "${!logitdecays[@]}";
do
    GPU="${gpus[i]}"
    logit_decay="${logitdecays[i]}"
    basedir="outdata/${schema}"
    datadir="${basedir}/${schema}"
    outdir="${outdir_base}/${schema}"
    mkdir -p $outdir
    checkpoint_dir="${checkpoint_dir_base}/${logit_decay}"

    echo "Schema $schema"
    echo "GPU $GPU"

    CMD="python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER --loss_type ${loss_type} --split ${split} --outdir ${outdir} --monitor_probs $monitor_probs --logit_decay ${logit_decay}" 
    CMD="nohup $CMD > $outdir/${logit_decay}.cout 2> $outdir/${logit_decay}.cerr &"
    CMD="CUDA_VISIBLE_DEVICES=$GPU $CMD"
    echo $CMD
    eval $CMD
done
