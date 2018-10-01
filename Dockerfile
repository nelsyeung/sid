FROM tensorflow/tensorflow:latest-gpu
WORKDIR /sid
RUN pip install tqdm
CMD ["python", "sid.py"]
