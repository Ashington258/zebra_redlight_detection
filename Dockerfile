# Use the official Python 3.8 image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# Copy the rest of the application code into the container
COPY . .

# Define the command to run the main script when the container starts
CMD ["python", "main.py"]