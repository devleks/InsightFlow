# InsightFlow - Deployment / DevOps Guide

This document provides guidance on deploying, managing, and updating the InsightFlow application.

## 1. Environments

Consider setting up distinct environments for development, staging (optional), and production to ensure stability and proper testing before changes go live.

*   **Development (Dev)**:
    *   Purpose: Local development and testing by developers.
    *   Setup: Typically run directly from a developer's machine (`streamlit run app.py`).
    *   Configuration: May use development-specific API keys (if available), local paths for data sources and vector stores.
*   **Staging (Optional)**:
    *   Purpose: Pre-production environment for testing new features and releases in a setup that closely mirrors production.
    *   Setup: Deployed to a server or platform similar to production.
    *   Configuration: Uses dedicated staging API keys, staging data sources, and a separate vector store.
*   **Production (Prod)**:
    *   Purpose: Live environment accessible to end-users.
    *   Setup: Deployed on a robust server or cloud platform.
    *   Configuration: Uses production API keys, production data sources, and a production vector store. Access should be restricted and monitored.

## 2. Deployment Options for Streamlit Applications

Streamlit applications can be deployed in various ways:

### a. Streamlit Community Cloud (formerly Streamlit Sharing)

*   **Pros**: Easy to deploy directly from a GitHub repository, free tier available, handles much of the infrastructure.
*   **Cons**: May have limitations on resources, compute, and customization for complex applications or private data.
*   **Process**:
    1.  Ensure your app is in a public or private GitHub repository.
    2.  Include a `requirements.txt` file.
    3.  Sign up for Streamlit Community Cloud and link your GitHub account.
    4.  Deploy the app from the dashboard by pointing to your repository and main Python file (`app.py`).
    5.  Manage secrets (API keys) via the Streamlit Community Cloud settings for your app.

### b. Docker Containerization

*   **Pros**: Highly portable, consistent environments, scalable, can be deployed on various container orchestration platforms (Kubernetes, Docker Swarm) or cloud services (AWS ECS, Google Cloud Run, Azure Container Instances).
*   **Process**:
    1.  Create a `Dockerfile`:
        ```dockerfile
        # Base Python image
        FROM python:3.9-slim

        # Set working directory
        WORKDIR /app

        # Install system dependencies if any (e.g., for certain parsers)
        # RUN apt-get update && apt-get install -y some-package

        # Copy requirements and install Python packages
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt

        # Copy the rest of the application code
        COPY . .

        # Expose the Streamlit port (default 8501)
        EXPOSE 8501

        # Command to run the Streamlit app
        # Use healthcheck for better monitoring if your platform supports it
        HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
        ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
        ```
    2.  Build the Docker image: `docker build -t insightflow-app .`
    3.  Run the container: `docker run -p 8501:8501 -v $(pwd)/.env:/app/.env -v $(pwd)/config.yaml:/app/config.yaml -v $(pwd)/your_data_path:/app/your_data_path -v $(pwd)/your_chroma_path:/app/your_chroma_path insightflow-app`
        *   Note: Volume mounts (`-v`) are crucial for `.env`, `config.yaml`, document data, and ChromaDB persistence.
    4.  Deploy to a container hosting service.

### c. Virtual Private Server (VPS) or Bare Metal

*   **Pros**: Full control over the environment.
*   **Cons**: Requires manual setup of the server, Python environment, reverse proxy, process management, and security.
*   **Process**:
    1.  Provision a server (e.g., AWS EC2, DigitalOcean Droplet, Linode).
    2.  Install Python, Git, and other necessary system dependencies.
    3.  Clone your application code.
    4.  Set up a virtual environment and install Python packages.
    5.  Configure `.env` and `config.yaml`.
    6.  **Process Manager**: Use a process manager like `systemd` or `supervisor` to run the Streamlit app as a service and ensure it restarts on failure.
        *   Example `systemd` service file (`insightflow.service`):
            ```ini
            [Unit]
            Description=InsightFlow Streamlit Application
            After=network.target

            [Service]
            User=your_user
            Group=your_group
            WorkingDirectory=/path/to/your/InsightFlow/app
            Environment="PYTHONUNBUFFERED=1"
            Environment="STREAMLIT_SERVER_PORT=8501"
            # Add other environment variables here or point to .env file if supervisor supports it
            ExecStart=/path/to/your/venv/bin/streamlit run app.py
            Restart=always

            [Install]
            WantedBy=multi-user.target
            ```
    7.  **Reverse Proxy**: Set up Nginx or Caddy as a reverse proxy to handle HTTPS (SSL/TLS), serve the app on standard ports (80/443), provide caching, and potentially load balancing.

## 3. CI/CD (Continuous Integration / Continuous Deployment)

*   **Purpose**: Automate testing and deployment of new changes.
*   **Tools**: GitHub Actions, GitLab CI/CD, Jenkins.
*   **Typical CI/CD Pipeline Steps**:
    1.  **Trigger**: On push to `develop` or `main` branch, or on PR merge.
    2.  **Checkout Code**: Get the latest version of the application.
    3.  **Set up Environment**: Install Python, dependencies.
    4.  **Linting/Static Analysis**: Check code quality (e.g., with Flake8, Black).
    5.  **Run Tests**: Execute unit and integration tests (e.g., `pytest`).
    6.  **(If tests pass) Build Artifacts**: E.g., build a Docker image.
    7.  **Push to Registry**: Push Docker image to a container registry (Docker Hub, AWS ECR, Google GCR).
    8.  **Deploy**: Deploy to the target environment (staging/production).
        *   This could involve updating a service on a VM, rolling out a new version in Kubernetes, or deploying to a PaaS.

## 4. Hosting Provider Details

This will depend on the chosen deployment option:

*   **Streamlit Community Cloud**: Managed by Streamlit.
*   **Cloud Providers (AWS, GCP, Azure)**: Offer various services suitable for hosting Streamlit apps (VMs, container services, serverless options).
*   **VPS Providers (DigitalOcean, Linode, Vultr)**: Provide virtual machines.

## 5. Secrets and Configuration Management

*   **Secrets (`.env` file)**: Contains API keys (OpenAI, Groq, etc.).
    *   **NEVER** commit `.env` to version control.
    *   Use `.gitignore` to exclude it.
    *   For CI/CD and production, inject secrets securely using platform-specific mechanisms (e.g., GitHub Actions secrets, AWS Secrets Manager, environment variables provided by the hosting service).
*   **Application Configuration (`config.yaml`)**: Contains non-secret application settings.
    *   Can be version-controlled if it doesn't contain secrets.
    *   For different environments (dev, staging, prod), you might have different versions of `config.yaml` or use environment variables to override specific settings within `config.yaml` at runtime.

## 6. Monitoring and Logging

*   **Streamlit Logs**: Streamlit outputs logs to the console where it's run. Process managers (`systemd`, `supervisor`) can capture these logs to files.
*   **Application Logs**: Implement more detailed logging within `app.py` using Python's `logging` module for errors, warnings, and important events.
*   **Performance Monitoring**: Tools like Prometheus/Grafana, Datadog, or cloud provider monitoring services can be used to track application performance, resource usage, and API call metrics.
*   **Health Checks**: If deploying with Docker or orchestration platforms, configure health check endpoints (Streamlit has a basic one at `/_stcore/health`).

This guide provides a starting point. The best deployment strategy will depend on your specific requirements, resources, and technical expertise.
