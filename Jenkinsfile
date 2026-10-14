// The same checks as .github/workflows/ci.yml, as a declarative Jenkins pipeline.
// STATUS: written but not run. Only the GitHub Actions pipeline is exercised in this project.
pipeline {
  agent { docker { image 'python:3.11-slim' args '-u root' } }
  environment { PIP_NO_CACHE_DIR = '1' }
  stages {
    stage('Install') {
      steps {
        sh 'apt-get update && apt-get install -y --no-install-recommends default-jre-headless'
        sh 'pip install -r requirements-dev.txt'
      }
    }
    stage('Lint') { steps { sh 'ruff check .' } }
    stage('Test') { steps { sh 'python -m pytest -q --junitxml=test-results.xml' } }
    stage('Regression check') { steps { sh 'python -m guard check' } }
    stage('Warehouse check') { steps { sh 'python -m warehouse verify' } }
  }
  post { always { junit 'test-results.xml' } }
}
