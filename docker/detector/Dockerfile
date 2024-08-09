ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.7.0-cudnn8

FROM vicoslab/echolib:${UBUNTU_VERSION} AS echolib-base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND noninteractive

# Get echolib
COPY --from=echolib-base /usr/local/ /usr/local/

ENV DIR /opt
WORKDIR $DIR

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git python3-dev python3-numpy-dev python3-pip libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 
RUN pip install future opencv_python torch==1.13.1 torchvision==0.14.1 pandas scikit-learn scikit-image tensorboard matplotlib segmentation_models_pytorch scipy tqdm

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} vicosdemo &&\
    useradd -l -u ${USER_ID} -g vicosdemo vicosdemo &&\
    mkdir /home/vicosdemo &&\
    chown -R ${USER_ID}:${GROUP_ID} /home/vicosdemo /opt

USER vicosdemo

ARG CEDIRNET_VERSION=11
RUN git clone --branch grasping --depth 1 https://DockerPull:glpat-scfQzKpmYhGz1wi3nXDD@gitlab.fri.uni-lj.si/vicos/divid/cedirnet-dev.git
RUN cd cedirnet-dev

ARG MODEL_VERSION="depth_large"
COPY --chown=vicosdemo:vicosdemo center_model.pth /opt/center_model.pth
COPY --chown=vicosdemo:vicosdemo ${MODEL_VERSION}.pth /opt/model.pth
COPY --chown=vicosdemo:vicosdemo configs/${MODEL_VERSION}.py /opt/cedirnet-dev/src/config/model_args.py

ENTRYPOINT [ "python3", "/opt/cedirnet-dev/tools/competition/infer_docker.py" ]