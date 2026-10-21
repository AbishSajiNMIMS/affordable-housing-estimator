#!/usr/bin/env bash
# Delete everything deploy_ecs.sh created, so there are no ongoing charges:
# the CloudFormation stack (load balancer, ECS service/cluster, EFS with the
# live data, VPC, roles, log group) and the ECR repository with its images.
#
# Usage:
#   export AWS_REGION=ap-south-1
#   ./scripts/cleanup_ecs.sh                 # stack + ECR repository
#   DELETE_SECRETS=1 ./scripts/cleanup_ecs.sh   # also delete the secrets
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION first}"
STACK_NAME="${STACK_NAME:-housing-estimator}"
ECR_REPOSITORY="${ECR_REPOSITORY:-affordable-housing-estimator}"
PREFIX="${SECRET_PREFIX:-housing-estimator}"

read -r -p "Delete stack '${STACK_NAME}' and ECR repo '${ECR_REPOSITORY}' in ${AWS_REGION}? Type 'yes': " ok
[ "$ok" = "yes" ] || { echo "Cancelled."; exit 1; }

echo "Deleting CloudFormation stack ${STACK_NAME} (takes 5-10 minutes)..."
aws cloudformation delete-stack --region "$AWS_REGION" --stack-name "$STACK_NAME"
aws cloudformation wait stack-delete-complete --region "$AWS_REGION" --stack-name "$STACK_NAME"

echo "Deleting ECR repository ${ECR_REPOSITORY} and its images..."
aws ecr delete-repository --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" --force >/dev/null 2>&1 \
  || echo "  (repository not found, skipping)"

if [ "${DELETE_SECRETS:-0}" = "1" ]; then
  for s in "${PREFIX}/scrappa-api-key" "${PREFIX}/refresh-token" "${PREFIX}/google-maps-key"; do
    aws secretsmanager delete-secret --region "$AWS_REGION" --secret-id "$s" \
      --force-delete-without-recovery >/dev/null 2>&1 && echo "Deleted secret $s" || true
  done
fi

echo "Done. Check the AWS Billing console in a day to confirm nothing is still running."
