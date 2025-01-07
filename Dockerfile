# Use a base image with Python and TensorFlow installed
FROM tensorflow/tensorflow:2.11.0-py3

# Set the working directory inside the container
WORKDIR /app

# Copy your Flask application to the container
COPY . /app

# Install any required Python dependencies
RUN pip install -r requirements.txt

# Expose the Flask app port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
