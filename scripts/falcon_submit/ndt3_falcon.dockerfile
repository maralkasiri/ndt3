# NDT3 dockerfiles for FALCON challenge
# Smoketest through benchmark start pack ./test_docker_local.sh --docker-name ndt3_smoke
# Submit through EvalAI CLI

# Need devel for flash attn
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel # This is the codebase pytorch version, but using updated image for python 3.11
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

RUN /bin/bash -c "python3 -m pip install falcon_challenge --upgrade"
# ADD ./falcon_challenge falcon_challenge
ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH = "/tmp/ground_truth.pkl"

# Users should install additional decoder-specific dependencies here.
RUN apt-get update && \
    apt-get install -y git
RUN pwd

# Copy local codebase to `context_general_bci` and pip install -e .
COPY context_general_bci /context_general_bci/
COPY setup.py /context_general_bci/setup.py
WORKDIR /context_general_bci
RUN python3 -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install flash-attn --no-build-isolation
WORKDIR /

ENV EVALUATION_LOC remote
# Add files from local context into Docker image
# Note local context reference is the working dir by default, see https://docs.docker.com/engine/reference/commandline/build/

# Add ckpt
# Note that Docker cannot easily import across symlinks, make sure data is not symlinked

# 10/3/24 ckpts (V5) -- CHANGE ARGS HERE
# ENV SPLIT "h1"
# ENV SPLIT "m1"
ENV SPLIT "m2"

# ENV MODEL "scratch"
# ENV MODEL "base_45m_200h"
ENV MODEL "big_350m_200h"


ENV NORM_PATH "data/norm.pt"
ENV CKPT_ROOT_DIR "local_data/ndt3_falcon_ckpts/$SPLIT/$MODEL"

ADD ./local_data/ndt3_${SPLIT}_norm.pt data/norm.pt
ENV CONFIG_STEM v5/tune/falcon_${SPLIT}/${MODEL}_100


# H1
# Avoid public submission of 2kh model, which bleeds test individual

## Scratch
# Best is best, but significantly overestimates perf of other seeds (by 0.1 - 0.2 R2). This one may be worth submitting multiply.
# ADD $CKPT_ROOT_DIR/l7f27hdh/checkpoints/val_kinematic_r2-epoch\=678-val_kinematic_r2\=0.5764-val_loss\=0.5331.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/5vsay5ep/checkpoints/val_kinematic_r2-epoch\=525-val_kinematic_r2\=0.5753-val_loss\=0.5369.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/8h6z1fw9/checkpoints/val_kinematic_r2-epoch\=418-val_kinematic_r2\=0.5254-val_loss\=0.5367.ckpt data/decoder.pth

# ## 45M 200h
# ADD $CKPT_ROOT_DIR/1ycqi4iy/checkpoints/val_kinematic_r2-epoch\=166-val_kinematic_r2\=0.5829-val_loss\=0.5352.ckpt  data/decoder.pth
# ADD $CKPT_ROOT_DIR/bbzh3cb9/checkpoints/val_kinematic_r2-epoch\=166-val_kinematic_r2\=0.5689-val_loss\=0.5352.ckpt   data/decoder.pth
# ADD $CKPT_ROOT_DIR/54fc84ke/checkpoints/val_kinematic_r2-epoch\=143-val_kinematic_r2\=0.5749-val_loss\=0.5335.ckpt  data/decoder.pth

# ## 350M 200h # Processings
# ADD $CKPT_ROOT_DIR/whguhao7/checkpoints/val_kinematic_r2-epoch\=188-val_kinematic_r2\=0.5989-val_loss\=0.5485.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/muhgvgd0/checkpoints/val_kinematic_r2-epoch\=265-val_kinematic_r2\=0.5810-val_loss\=0.6194.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/zfu3rtt4/checkpoints/val_kinematic_r2-epoch\=277-val_kinematic_r2\=0.5902-val_loss\=0.6039.ckpt data/decoder.pth

# M1

## Scratch
# ADD $CKPT_ROOT_DIR/7g2muyop/checkpoints/val_kinematic_r2-epoch=528-val_kinematic_r2=0.7904-val_loss=0.5305.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/34tr743u/checkpoints/val_kinematic_r2-epoch=504-val_kinematic_r2=0.7851-val_loss=0.5250.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/mjlu1ha6/checkpoints/val_kinematic_r2-epoch=842-val_kinematic_r2=0.7843-val_loss=0.5275.ckpt data/decoder.pth

# ## 45M 200h
# ADD $CKPT_ROOT_DIR/45471yb3/checkpoints/val_kinematic_r2-epoch\=183-val_kinematic_r2\=0.7933-val_loss\=0.5337.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/ijkmj764/checkpoints/val_kinematic_r2-epoch\=79-val_kinematic_r2\=0.7875-val_loss\=0.5260.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/0t2t2wy7/checkpoints/val_kinematic_r2-epoch\=56-val_kinematic_r2\=0.7908-val_loss\=0.5275.ckpt data/decoder.pth

# ## 350M 200h
# ADD $CKPT_ROOT_DIR/123g06oh/checkpoints/val_kinematic_r2-epoch=416-val_kinematic_r2=0.7924-val_loss=0.7357.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/38chrvmf/checkpoints/val_kinematic_r2-epoch=266-val_kinematic_r2=0.7908-val_loss=0.5619.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/ijkv0rf6/checkpoints/val_kinematic_r2-epoch=273-val_kinematic_r2=0.7849-val_loss=0.5327.ckpt data/decoder.pth


# M2

## Scratch
# ADD $CKPT_ROOT_DIR/n82za43t/checkpoints/val_kinematic_r2-epoch=363-val_kinematic_r2=0.5839-val_loss=0.2033.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/c0ohvrou/checkpoints/val_kinematic_r2-epoch=366-val_kinematic_r2=0.5820-val_loss=0.2038.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/pf0x8g1q/checkpoints/val_kinematic_r2-epoch=380-val_kinematic_r2=0.5783-val_loss=0.2048.ckpt data/decoder.pth

# ## 45M 200h

# ADD $CKPT_ROOT_DIR/idwjiuzp/checkpoints/val_kinematic_r2-epoch\=90-val_kinematic_r2\=0.5662-val_loss\=0.1991.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/e6vpi57n/checkpoints/val_kinematic_r2-epoch\=277-val_kinematic_r2\=0.5673-val_loss\=0.2092.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/5tijo9yd/checkpoints/val_kinematic_r2-epoch\=146-val_kinematic_r2\=0.5744-val_loss\=0.2011.ckpt data/decoder.pth


# ## 350M 200h # TODO submit
# ADD $CKPT_ROOT_DIR/4ry0sk23/checkpoints/val_kinematic_r2-epoch=85-val_kinematic_r2=0.5751-val_loss=0.1993.ckpt data/decoder.pth
# ADD $CKPT_ROOT_DIR/osednzui/checkpoints/val_kinematic_r2-epoch=143-val_kinematic_r2=0.5747-val_loss=0.2052.ckpt data/decoder.pth
ADD $CKPT_ROOT_DIR/zpcddqmz/checkpoints/val_kinematic_r2-epoch=134-val_kinematic_r2=0.5707-val_loss=0.2066.ckpt data/decoder.pth


# Add runfile
RUN pwd
ADD ./scripts/falcon_submit/ndt3_falcon_runner.py decode.py

ENV BATCH_SIZE 8
ENV PHASE "test"
RUN pwd
# ENV PHASE "minival"

# Make sure this matches the mounted data volume path. Generally leave as is.
ENV EVAL_DATA_PATH "/dataset/evaluation_data"

# CMD specifies a default command to run when the container is launched.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py \
    --evaluation $EVALUATION_LOC \
    --model-path data/decoder.pth \
    --config-stem $CONFIG_STEM \
    --norm-path $NORM_PATH \
    --split $SPLIT \
    --batch-size $BATCH_SIZE \
    --phase $PHASE"]
