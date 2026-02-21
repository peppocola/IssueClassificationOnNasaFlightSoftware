# Use Python 3.11.6 as specified in the requirements
FROM python:3.11.6-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Clone git submodules directly (since .git directory is not copied)
# Remove empty submodule directory first if it exists
RUN rm -rf externals/sklearn-cls-report2excel && \
    git clone https://github.com/seanswyi/sklearn-cls-report2excel.git externals/sklearn-cls-report2excel

# Create necessary directories
RUN mkdir -p output logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Add startup script for better visibility
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Default command (can be overridden)
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "main.py"]
