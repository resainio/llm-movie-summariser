FROM python:3.10

WORKDIR /llm-movie-summariser

COPY ./requirements.txt /llm-movie-summariser/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /llm-movie-summariser/requirements.txt

COPY ./app /llm-movie-summariser/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]