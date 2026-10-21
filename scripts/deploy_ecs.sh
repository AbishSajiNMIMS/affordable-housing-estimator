#!/usr/bin/env bash
# Build the Docker image, push it to Amazon ECR and deploy/update the
# CloudFormation stack (VPC, load balancer, ECS Fargate service, EFS).
#
# Usage:
#   export AWS_REGION=ap-south-1                  # any region with Fargate
#   ./scripts/deploy_ecs.sh
#
# Optional environment variables:
#   STACK_NAME        (default: housing-estimator)
#   ECR_REPOSITORY    (default: affordable-housing-estimator)
#   IMAGE_TAG         (default: current git commit, or a timestamp)
#   SCRAPPA_SECRET_ARN, REFRESH_TOKEN_SECRET_ARN, GOOGLE_MAPS_SECRET_ARN  (from scripts/create_secrets.sh)
#   AUTO_REFRESH_HOURS (default: 0 = off), LIVE_CITIES (default: bangalore)
#   CERTIFICATE_ARN   (optional ACM certificate for HTTPS)
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, e.g. export AWS_REGION=ap-south-1}"
STACK_NAME="${STACK_NAME:-housing-estimator}"
ECR_REPOSITORY="${ECR_REPOSITORY:-affordable-housing-estimator}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
SKIP_BUILD="${SKIP_BUILD:-0}"

cd "$(dirname "$0")/.."

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"
echo "Account ${ACCOUNT_ID}, region ${AWS_REGION}, image ${IMAGE_URI}"

if ! aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$ECR_REPOSITORY" >/dev/null 2>&1; then
  echo "Creating ECR repository ${ECR_REPOSITORY}"
  aws ecr create-repository --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" \
    --image-scanning-configuration scanOnPush=true >/dev/null
  # Keep only the 10 most recent images to limit storage cost.
  aws ecr put-lifecycle-policy --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" \
    --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"keep last 10","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' >/dev/null
fi

if [ "$SKIP_BUILD" != "1" ]; then
  aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
  # Fargate runs x86_64 in this template; --platform matters on Apple Silicon Macs.
  docker build --platform linux/amd64 -t "$IMAGE_URI" .
  docker push "$IMAGE_URI"
fi

# Only pass optional settings that are set; on updates, unset ones keep
# their previous values (so re-running without the secret ARNs is safe).
PARAMS=("ImageUri=${IMAGE_URI}")
[ -n "${LIVE_CITIES:-}" ] && PARAMS+=("LiveCities=${LIVE_CITIES}")
[ -n "${AUTO_REFRESH_HOURS:-}" ] && PARAMS+=("AutoRefreshHours=${AUTO_REFRESH_HOURS}")
[ -n "${SCRAPPA_SECRET_ARN:-}" ] && PARAMS+=("ScrappaApiKeySecretArn=${SCRAPPA_SECRET_ARN}")
[ -n "${REFRESH_TOKEN_SECRET_ARN:-}" ] && PARAMS+=("RefreshTokenSecretArn=${REFRESH_TOKEN_SECRET_ARN}")
[ -n "${GOOGLE_MAPS_SECRET_ARN:-}" ] && PARAMS+=("GoogleMapsKeySecretArn=${GOOGLE_MAPS_SECRET_ARN}")
[ -n "${CERTIFICATE_ARN:-}" ] && PARAMS+=("CertificateArn=${CERTIFICATE_ARN}")

aws cloudformation deploy \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --template-file infra/ecs-fargate.yaml \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides "${PARAMS[@]}"

URL="$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='WebsiteUrl'].OutputValue" --output text)"
echo
echo "Deployed. Website: ${URL}"
echo "The first start can take 2-3 minutes before the load balancer reports the task healthy."
