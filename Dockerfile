FROM python:3.11.5-slim

WORKDIR /flask_app

COPY . /flask_app/

RUN pip install -r requirements.txt

EXPOSE 8000


ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:8000", "app:app" ]