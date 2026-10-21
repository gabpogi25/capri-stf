# Download and extract COLMAP for Windows automatically

$colmap_url = "https://github.com/colmap/colmap/releases/download/3.9.1/COLMAP-3.9.1-windows-cuda.zip"
$zip_path = "COLMAP.zip"
$extract_dir = "colmap"

Write-Host "Downloading COLMAP 3.9.1 (CUDA enabled)..."
Invoke-WebRequest -Uri $colmap_url -OutFile $zip_path

Write-Host "Extracting COLMAP..."
Expand-Archive -Path $zip_path -DestinationPath $extract_dir -Force

Write-Host "Cleaning up zip file..."
Remove-Item $zip_path

Write-Host "Done! COLMAP is ready in the .\colmap folder."
