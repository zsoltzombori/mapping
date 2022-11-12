# parent: exp91
# comparing lprp with seq_prp
# changes:  less epochs, larger lr, optimizer, 
#           added alpha and opt_steps
#           monitor_probs = 1 !!

exp=911
epochs=10
batch_size=20
neg_weight=0.0
num_layers=4
d_model=128
lr=0.01
lr_decay_steps=30
optimizer=adamax
beta1=0.3
beta2=0.9
CHAR_TOKENIZER=0
loss_type="seq_prp"
split="0.7,0.15,0.15"
monitor_probs=1
opt_steps=10
alpha=1.05

# schemas=( cmt_renamed cmt_denormalized cmt_mixed cmt_naive_ci cmt_structured_ci cmt_structured )
# gpus=( 0 1 2 2 3 3 )
# schemas=( cmt_mixed cmt_renamed cmt_denormalized cmt_structured )
# gpus=( 0 1 2 3 )

schemas=( cmt_renamed cmt_denormalized cmt_structured )
gpus=( 1 2 3 )

outdir_base="out/exp${exp}"
checkpoint_dir_base="checkpoints/exp${exp}"

for i in "${!schemas[@]}";
do
    GPU="${gpus[i]}"
    schema="${schemas[i]}"
    basedir="outdata/${schema}"
    datadir="${basedir}/${schema}"
    outdir="${outdir_base}/${schema}"
    mkdir -p $outdir
    checkpoint_dir="${checkpoint_dir_base}/${schema}"

    echo "Schema $schema"
    echo "GPU $GPU"

    CMD="python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER --loss_type ${loss_type} --split ${split} --outdir ${outdir} --monitor_probs $monitor_probs --opt_steps $opt_steps --multiplier $alpha"   
    CMD="nohup $CMD > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
    CMD="CUDA_VISIBLE_DEVICES=$GPU $CMD"
    echo $CMD
    eval $CMD
done
