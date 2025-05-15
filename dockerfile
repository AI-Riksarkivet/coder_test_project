# Dockerfile

# Use an official Python runtime as a parent image
# We choose a -slim variant for a smaller image size, compatible with python>=3.12
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install build tools that might be needed by some Python packages (e.g., for compiling C extensions)
# and then install uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir uv

# Copy the pyproject.toml file first to leverage Docker layer caching for dependencies
COPY pyproject.toml .

# Install Python dependencies using uv from pyproject.toml
# The command `uv pip install .` will install the current project (defined by pyproject.toml)
# and its dependencies listed in the [project.dependencies] section.
# Using --no-cache to reduce image size.
# Note: Ensure the versions in pyproject.toml are available.
# The futuristic versions (e.g., dask>=2025.3.0) might cause issues if not actually released.
RUN uv pip install --no-cache .

# Copy the rest of the application code
# For this example, it's just main.py
COPY main.py .
# If you had other Python modules or a package structure, you'd copy them here.
# For example: COPY src/ .

# Command to run the application
# This will execute 'python main.py' when the container starts
CMD ["python", "main.py"]