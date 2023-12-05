FROM python:3.7
RUN pip install mlflow gunicorn
EXPOSE 8000
COPY mlflow_server.py /app/mlflow_server.py
COPY wsgi.py /app/wsgi.py
WORKDIR /app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "wsgi:app"]
