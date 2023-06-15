FROM python:3.10.11-bullseye
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
