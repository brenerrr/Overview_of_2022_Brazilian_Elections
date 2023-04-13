FROM continuumio/miniconda3
COPY requirements.txt /tmp/
COPY ./app /app
WORKDIR "/app"
RUN ls
RUN pip install -r /tmp/requirements.txt

# EXPOSE 8050/tcp

# ENTRYPOINT [ "python3" ]

# CMD [ "app.py" ]

EXPOSE 80

CMD ["python", "app.py"]