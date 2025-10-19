# IRIS ML Pipeline with CI/CD and DVC

## Overview
This repository contains an end-to-end **machine learning pipeline** for the IRIS dataset, including:

- Data validation
- Model evaluation
- Continuous Integration (CI) and Continuous Deployment (CD)
- DVC-based data and model versioning
- Automated reporting using CML

The pipeline is designed to be **reproducible, automated, and cloud-ready**.

---

## Features

- **Two Branches**: `dev` for development, `main` for production
- **Unit Tests**: Implemented using `pytest` for:
  - Data validation (`test_data_validation.py`)
  - Model evaluation (`evaluate.py`)
- **CI/CD**: Configured with GitHub Actions
  - Runs on push to `dev` or `main`
  - Runs on PR merges
  - Pulls model and data from DVC
- **CML Integration**: Automatically posts evaluation reports as comments on PRs
- **Cloud Setup**: Demonstrated using Google Cloud Platform (GCP) VM

---
CI/CD Workflow

GitHub Actions workflows are located in .github/workflows/.

Triggers:

push to dev or main

pull_request targeting main

Actions performed:

Install dependencies

Pull DVC data and models

Run unit tests

Execute evaluation

Post results using CML

---

Learnings

CI/CD integration for ML pipelines

DVC for reproducible experiments

Automated testing and reporting

Cloud-based ML pipeline execution
## Directory Structure

