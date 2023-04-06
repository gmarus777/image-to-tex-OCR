# Starting from an official AWS image
# Keep any dependencies and versions in this file aligned with the environment.yml and Makefile
FROM public.ecr.aws/lambda/python:3.9

# Install Python dependencies
COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip==22.0.4
RUN pip install -r production_reqs.txt

# Copy only the relevant directories and files
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY Tokenizer/ ./Tokenizer
COPY server_api/ ./api

CMD ["api.handler"]