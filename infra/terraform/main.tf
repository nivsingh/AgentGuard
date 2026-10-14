# AWS deployment of the AgentGuard demo app: an ECR repository for the image, AWS App Runner to serve it,
# and Secrets Manager for the model API key.
#
# STATUS: written but never applied or validated. The sandbox this project was built in could not download
# Terraform. Run `terraform init && terraform validate && terraform plan` before trusting it.
#
#   terraform apply -var 'image_tag=latest'      # after pushing the image (see .github/workflows/deploy.yml)

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "agentguard"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "anthropic_api_key" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Optional. Leave empty to run the app with the simulated model."
}

resource "aws_ecr_repository" "app" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_secretsmanager_secret" "anthropic" {
  count = var.anthropic_api_key == "" ? 0 : 1
  name  = "${var.name}/anthropic-api-key"
}

resource "aws_secretsmanager_secret_version" "anthropic" {
  count         = var.anthropic_api_key == "" ? 0 : 1
  secret_id     = aws_secretsmanager_secret.anthropic[0].id
  secret_string = var.anthropic_api_key
}

# App Runner pulls the image from ECR with this role.
resource "aws_iam_role" "ecr_access" {
  name = "${var.name}-apprunner-ecr"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "build.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_access" {
  role       = aws_iam_role.ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# The running service reads the secret with this role.
resource "aws_iam_role" "instance" {
  name = "${var.name}-apprunner-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "tasks.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "read_secret" {
  count = var.anthropic_api_key == "" ? 0 : 1
  name  = "read-anthropic-secret"
  role  = aws_iam_role.instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.anthropic[0].arn
    }]
  })
}

resource "aws_apprunner_service" "app" {
  service_name = var.name

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.ecr_access.arn
    }
    image_repository {
      image_identifier      = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"
      image_configuration {
        port = "8501"
        runtime_environment_secrets = var.anthropic_api_key == "" ? {} : {
          ANTHROPIC_API_KEY = aws_secretsmanager_secret.anthropic[0].arn
        }
      }
    }
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu               = "1024"
    memory            = "2048"
    instance_role_arn = aws_iam_role.instance.arn
  }

  health_check_configuration {
    protocol = "HTTP"
    path     = "/_stcore/health"
  }
}

output "app_url" {
  value = "https://${aws_apprunner_service.app.service_url}"
}

output "ecr_repository" {
  value = aws_ecr_repository.app.repository_url
}
