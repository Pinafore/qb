FROM python:3.5
ENV PYTHONUNBUFFERED 1

RUN mkdir -p /app/src
WORKDIR /app/src
ADD requirements.txt /app/src/
RUN pip install -r requirements.txt
ADD . /app/src/