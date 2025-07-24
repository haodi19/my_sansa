gpu=$1  # 4
port=$2  # 1234
dataset=$3  # pascal/coco
exp_name=split$4  # 0/1/2/3
shot=$5  # 1/5
arch=$6  # FSSAM
net=$7  # small

if [ $shot -eq 1 ]; then
  postfix=batch
elif [ $shot -eq 5 ]; then
  postfix=5s_batch
else
  echo "Only 1 and 5 shot are supported"
  exit 1
fi

# exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}_sem_384_24_adapter_newtrans2
exp_dir=exp/${dataset}/${arch}/${exp_name}/tmp
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
# config=config/${dataset}/${net}/2.1/${dataset}_${exp_name}_${net}_${postfix}.yaml
config=/hdd0/ljn/new_sam2/my_fssam/config/coco/large/2.1/coco_split0_large_5s_batch.yaml
# config=config/coco/large/2.1/coco_split0_large_batch_sansa.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")

echo ${arch}
echo ${config}

# if [ $shot -eq 1 ]; then
#   mkdir -p priors/${dataset}/${net}/${ver_dino}/${exp_name}/1shot
# elif [ $shot -eq 5 ]; then
#   mkdir -p priors/${dataset}/${net}/${ver_dino}/${exp_name}/5shot
# else
#   echo "Only 1 and 5 shot are supported"
#   exit 1
# fi

python3 -m torch.distributed.launch --nproc_per_node=${gpu} --master_port=${port} train_multi_frame.py \
        --config=${config} \
        --arch=${arch} \
        # 2>&1 | tee ${result_dir}/train-$now.log
