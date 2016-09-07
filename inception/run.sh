NOWDIR=`dirname "$0"`
NOWDIR=`cd $NOWDIR; pwd`
export PYTHONPATH=${HOME}/workshops/FinalProject2/inception/

python $NOWDIR/inception/flowers_train.py \
    --train_dir=$NOWDIR/xent_log_0907 \
    --data_dir=/home/zhangxuesen/workshops/data/tfrecord \
    --initial_learning_rate=0.1 \
    --input_queue_memory_factor=1 \
    --max_steps=35000 \
    --max_steps=100 \
    --num_epochs_per_decay=30 \
    --learning_rate_decay_factor=0.26 \
    --batch_size=128 \
    --image_size=32 \
    --num_readers=1 \
    --num_preprocess_threads=4 \
    --xent=xent \
#    --pretrained_model_checkpoint_path=$NOWDIR/log_0906/model.ckpt-34999 \
