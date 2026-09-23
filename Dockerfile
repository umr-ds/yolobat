FROM ultralytics/ultralytics:8.4.21

RUN apt update && \
    apt install -y --no-install-recommends \
    ffmpeg \
    build-essential

# The base image ships Ultralytics as an editable install at /ultralytics, but at
# a different commit. Move it to the one YOLObat was developed against and apply
# our changes. Own layer, so editing this repository does not redo it.
ARG ULTRALYTICS_COMMIT=7710ef05dc56dfe61b20a7537f17078db2b7170b
COPY ultralytics-yolobat.patch /tmp/ultralytics-yolobat.patch
RUN git -C /ultralytics fetch -q --depth 1 origin ${ULTRALYTICS_COMMIT} && \
    git -C /ultralytics checkout -q -f FETCH_HEAD && \
    sed -i 's/"opencv-python>=/"opencv-python-headless>=/' /ultralytics/pyproject.toml && \
    git -C /ultralytics apply /tmp/ultralytics-yolobat.patch && \
    rm /tmp/ultralytics-yolobat.patch

COPY . /yolobat-train
WORKDIR /yolobat-train
RUN pip install -r requirements.txt
RUN pip install ipywidgets jupyterlab

ENV PYTHONPATH="/yolobat-train/:/yolobat-train/trainer:/yolobat-train/data:/ultralytics/ultralytics"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_cupti/lib"
