# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Firefox
RUN apt-get update && apt-get install -y wget xvfb firefox-esr

# Install Geckodriver
RUN GECKODRIVER_VERSION=$(wget --no-verbose -O - "https://api.github.com/repos/mozilla/geckodriver/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/') \
    && wget -q --no-verbose -O geckodriver.tar.gz "https://github.com/mozilla/geckodriver/releases/download/$GECKODRIVER_VERSION/geckodriver-$GECKODRIVER_VERSION-linux64.tar.gz" \
    && tar -xzf geckodriver.tar.gz -C /usr/local/bin \
    && rm geckodriver.tar.gz

# Install Selenium
RUN pip install selenium

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Run when the container launches
CMD ["python", "script.py"]

