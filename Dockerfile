FROM ubuntu
RUN apt-get update -y

RUN apt-get install -y python3-pip
RUN apt-get install -y python3

RUN pip3 install --upgrade pip


WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./model /code/model
COPY ./preprocessing /code/preprocessing
COPY ./test ./code/test
COPY ./app.py /code/app.py
COPY ./saved_objects.pkl /code/saved_objects.pkl

RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
