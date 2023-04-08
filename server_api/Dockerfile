
FROM python:3.9-buster



RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY production_reqs.txt ./requirements.txt
RUN pip install --upgrade pip==22.0.4
RUN pip install -r requirements.txt

# Copy only the relevant directories and files
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY Tokenizer/ ./Tokenizer
COPY server_api/ ./server_api
COPY streamlit_app/ ./streamlit_app
COPY Models_Parameters_Log/scripted_model1.pt ./Models_Parameters_Log/scripted_model1.pt

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD ["uvicorn", "server_api.app:app", "--host", "0.0.0.0", "--port", "8000"]