# Get ACR credentials
$ACR_NAME = "fraudacr12345"
$ACR_RG = "fraud-rg"
$ACR_LOGIN_SERVER = az acr show -n $ACR_NAME -g $ACR_RG --query loginServer -o tsv
$ACR_USERNAME = az acr credential show -n $ACR_NAME -g $ACR_RG --query username -o tsv
$ACR_PASSWORD = az acr credential show -n $ACR_NAME -g $ACR_RG --query passwords[0].value -o tsv

# Create Kubernetes secret in the "default" namespace
kubectl create secret docker-registry acr-secret `
  --docker-server=$ACR_LOGIN_SERVER `
  --docker-username=$ACR_USERNAME `
  --docker-password=$ACR_PASSWORD