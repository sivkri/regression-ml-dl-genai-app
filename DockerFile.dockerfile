# Base image with Python and pip
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all app files to the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit-specific config (optional)
ENV STREAMLIT_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
