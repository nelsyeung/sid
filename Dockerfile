FROM tensorflow/tensorflow:latest-gpu
WORKDIR /sid
CMD ["python", "sid.py"]
