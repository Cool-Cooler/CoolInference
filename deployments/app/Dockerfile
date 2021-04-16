FROM python:3.7
ADD app/ /app
WORKDIR /app
RUN apt-get update
RUN apt-get install -y python3-opencv
RUN pip install -r requirements.txt
RUN pip install pyyaml==5.1 pycocotools>=2.0.1
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html 
CMD python app.py
