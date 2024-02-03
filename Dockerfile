FROM ubuntu
RUN apt-get update -y

RUN apt-get install -y python3-pip
RUN apt-get install -y python3

RUN pip3 install --upgrade pip


WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./utils /code/utils
COPY ./preprocessing /code/preprocessing
COPY ./app.py /code/app.py

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
