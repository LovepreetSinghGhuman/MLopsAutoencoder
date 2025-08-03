param(
    [string]$Namespace = "default"
)

Write-Host "Listing pods in namespace: $Namespace"
kubectl get pods -n $Namespace

Write-Host "`nFetching logs for all pods in namespace: $Namespace"
$pods = kubectl get pods -n $Namespace --no-headers | ForEach-Object { ($_ -split '\s+')[0] }
foreach ($pod in $pods) {
    Write-Host "=============================="
    Write-Host "Logs for pod: $pod"
    $containers = kubectl get pod $pod -n $Namespace -o jsonpath='{.spec.containers[*].name}'
    foreach ($container in $containers -split " ") {
        Write-Host "--- Container: $container ---"
        kubectl logs $pod -c $container -n $Namespace
        Write-Host ""
    }
}