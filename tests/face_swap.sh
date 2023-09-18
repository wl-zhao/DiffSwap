CKPT=checkpoints/diffswap.pth
PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

PYTHONPATH=./:$PYTHONPATH python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT tests/faceswap_portrait.py $CKPT --save_img True