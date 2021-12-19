FROM python:3.8.12

COPY . /ml_flask_api
WORKDIR /ml_flask_api

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 1000

ENTRYPOINT ["python3"]
CMD ["api.py"]