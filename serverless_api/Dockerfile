# Starting from an official AWS image
# Keep any dependencies and versions in this file aligned with the environment.yml and Makefile
FROM public.ecr.aws/lambda/python:3.9

# Install Python dependencies
COPY production_reqs.txt ./requirements.txt
RUN pip install --upgrade pip==22.0.4
RUN pip install -r requirements.txt

# Copy only the relevant directories and files
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY Tokenizer/ ./Tokenizer
COPY serverless_api/ ./serverless_api
COPY Models_Parameters_Log/scripted_model1.pt ./Models_Parameters_Log/scripted_model1.pt

CMD ["serverless_api/app.handler"]