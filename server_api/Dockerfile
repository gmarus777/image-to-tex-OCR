
# Starting from an official AWS image
FROM amazon/aws-lambda-python:3.6


# Install Python dependencies
COPY production_reqs.txt ./requirements.txt
RUN pip install --upgrade pip==22.0.4
RUN pip install -r requirements.txt

# Copy only the relevant directories and files
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY Tokenizer/ ./Tokenizer
COPY serverless_api/ ./serverless_api
COPY streamlit_app/ ./streamlit_app
COPY Models_Parameters_Log/scripted_model1.pt ./Models_Parameters_Log/scripted_model1.pt

CMD ["app.handler"]