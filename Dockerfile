FROM tensorflow/tensorflow:latest-gpu

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt install -y git \
    vim  \
    libgl1-mesa-glx

COPY requirements.txt .
    
RUN pip install --upgrade pip

RUN pip3 install -r requirements.txt

RUN mkdir -p /root/.cache/pydrive2fs/710796635688-iivsgbgsb6uv1fap6635dhvuei09o66c.apps.googleusercontent.com

COPY default.json /root/.cache/pydrive2fs/710796635688-iivsgbgsb6uv1fap6635dhvuei09o66c.apps.googleusercontent.com 

WORKDIR /steak_image_segmentation

ENV SM_FRAMEWORK=tf.keras

COPY . .


