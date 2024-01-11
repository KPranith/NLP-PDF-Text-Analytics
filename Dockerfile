# Use an official Python runtime as a parent image
FROM python:3

# Install git
RUN apt-get update && apt-get install -y git

# Clone the Git repository into the container
RUN git clone https://github.com/KPranith/NLP-PDF-Text-Analytics.git

# Set the working directory to /app
WORKDIR /NLP-PDF-Text-Analytics

# RUN pip3 install -r /NLP-PDF-Text-Analytics/requirements.txt

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
ENTRYPOINT ["tail", "-f", "/dev/null"]