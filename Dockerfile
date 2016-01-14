FROM python:3

RUN mkdir /qb
WORKDIR /qb
ADD requirements.txt /qb/requirements.txt
RUN pip install -r requirements.txt
RUN python setup.py download

ADD . /qb
