# Use an official Python runtime as a parent image
FROM python:3.9 AS base

LABEL maintainer="Your Name praneeth.kunala@gmail.com"
LABEL version="1.0"
LABEL description="Docker image for NLP PDF Text Analytics- Python 3.9" \
    org.opencontainers.image.title="NLP-PDF-Text-Analytics" \
    org.opencontainers.image.description="Docker image for NLP PDF Text Analytics- Python 3.9" \
    org.opencontainers.image.version="1.0" \
    org.opencontainers.image.authors="Your Name praneeth.kunala@gmail.com" \
    org.opencontainers.image.name="nlp-pdf-text-analytics"

# Install git
RUN apt-get update && apt-get install -y git

# Clone the Git repository into the container
RUN git clone https://github.com/KPranith/NLP-PDF-Text-Analytics.git

# Set the working directory to /app
WORKDIR /NLP-PDF-Text-Analytics

RUN pip3 install -r /NLP-PDF-Text-Analytics/requirements.txt

# Define environment variable
# ENV NAME World
# Add a volume mount to map the files directory
VOLUME ["/NLP-PDF-Text-Analytics/files"]
# Run app.py when the container launches
ENTRYPOINT ["tail", "-f", "/dev/null"]