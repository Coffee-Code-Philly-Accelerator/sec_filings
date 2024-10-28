FROM ubuntu:latest

# Install Firefox and prerequisites for running GUI
RUN apt-get update && apt-get install -y firefox dbus-x11

# Set up environment variables to enable GUI support
ENV DISPLAY=host.docker.internal:0

WORKDIR /usr/src/app

# Install Selenium
RUN apt-get install -y python3-pip
#RUN pip3 install selenium

# Copy the Selenium script into the container
COPY . .

# Command to run when starting the container
#CMD ["python3", "script.py"]

