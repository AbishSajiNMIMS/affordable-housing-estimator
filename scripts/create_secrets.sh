#!/usr/bin/env bash
# Store the optional Scrappa API key and admin refresh token in AWS Secrets
# Manager, and print the ARNs to pass to deploy_ecs.sh.
#
# Usage:
#   export AWS_REGION=ap-south-1
#   ./scripts/create_secrets.sh
# You will be prompted for the values (input is hidden, nothing is written to disk).
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION first}"
PREFIX="${SECRET_PREFIX:-housing-estimator}"

upsert() {  # name value
  local name="$1" value="$2" arn
  if arn="$(aws secretsmanager describe-secret --region "$AWS_REGION" --secret-id "$name" --query ARN --output text 2>/dev/null)"; then
    aws secretsmanager put-secret-value --region "$AWS_REGION" --secret-id "$name" --secret-string "$value" >/dev/null
  else
    arn="$(aws secretsmanager create-secret --region "$AWS_REGION" --name "$name" --secret-string "$value" --query ARN --output text)"
  fi
  echo "$arn"
}

read -r -s -p "Scrappa API key (leave empty to skip): " API_KEY; echo
read -r -s -p "Admin refresh token (leave empty to generate one): " TOKEN; echo
if [ -z "$TOKEN" ]; then
  TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
  echo "Generated admin refresh token: ${TOKEN}"
  echo "Save it in your password manager; you type it into the website's 'Refresh now' box."
fi

echo
if [ -n "$API_KEY" ]; then
  echo "export SCRAPPA_SECRET_ARN=$(upsert "${PREFIX}/scrappa-api-key" "$API_KEY")"
fi
echo "export REFRESH_TOKEN_SECRET_ARN=$(upsert "${PREFIX}/refresh-token" "$TOKEN")"
echo
echo "Copy the export line(s) above into your terminal, then run ./scripts/deploy_ecs.sh"
