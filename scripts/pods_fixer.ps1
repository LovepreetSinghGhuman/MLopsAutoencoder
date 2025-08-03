param(
    [string]$Namespace = "default"
)

Write-Host "[$(Get-Date -Format 'u')] Listing pods in namespace: $Namespace"
$podInfo = kubectl get pods -n $Namespace --no-headers | ForEach-Object {
    $fields = $_ -split '\s+'
    [PSCustomObject]@{
        Name = $fields[0]
        Ready = $fields[1]
        Status = $fields[2]
        Restarts = $fields[3]
        Age = $fields[4]
    }
}
$podInfo | Format-Table

Write-Host "`n[$(Get-Date -Format 'u')] Describing all pods in namespace: $Namespace"
foreach ($pod in $podInfo) {
    Write-Host "=============================="
    Write-Host "[$(Get-Date -Format 'u')] Describe for pod: $($pod.Name)"
    Write-Host "Status: $($pod.Status) | Ready: $($pod.Ready) | Restarts: $($pod.Restarts) | Age: $($pod.Age)"
    try {
        kubectl describe pod $($pod.Name) -n $Namespace
    } catch {
        Write-Host "Error describing pod $($pod.Name)"
    }
    Write-Host ""
}