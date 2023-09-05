FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install --no-cache -U pip
EXPOSE 80
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["App.py"]
