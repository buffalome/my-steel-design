# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# FROM python:3.10-slim
FROM python:3.8-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup. 
CMD panel serve steel_design.py --address 0.0.0.0 --port 8080 --allow-websocket-origin="*" 