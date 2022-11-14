#
FROM ubuntu:20.04

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && apt-get install -y python3.8 python3-dev python3-pip libsm6 libxext6 libxrender-dev ffmpeg vim software-properties-common tzdata tesseract-ocr

#
WORKDIR /app/oc-mykad

#
COPY . /app/oc-mykad

#
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt


ENV LANG C.UTF-8
ENV TESSDATA_PREFIX etc/traineddata/
ENV PYTHONUNBUFFERED=1
ENV OMP_THREAD_LIMIT=1

#
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:6000", "--workers", "4", "--threads", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]