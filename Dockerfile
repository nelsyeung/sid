FROM tensorflow/tensorflow:latest-gpu
WORKDIR /sid
RUN apt update && apt install -y python-tk
RUN pip install tqdm
CMD ["python", "sid.py"]
