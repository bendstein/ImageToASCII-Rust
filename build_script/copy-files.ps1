# Define the array of files and directories to be copied
$itemsToCopy = @(
    "config.toml",
    "assets"
)

# Check if the destination directory is provided as a command line argument
if ($args.Count -eq 0) {
    Write-Host "Please provide a destination directory."
    exit 1
}

# Get the destination directory from the command line argument
$destinationDir = $args[0]

# Ensure the destination directory exists
if (-not (Test-Path $destinationDir)) {
    Write-Host "The destination directory does not exist. Creating it now..."
    New-Item -ItemType Directory -Path $destinationDir
}

# Copy each item in the array to the destination directory
foreach ($item in $itemsToCopy) {
    if (Test-Path $item) {
        Copy-Item -Path $item -Destination $destinationDir -Recurse -Force
        Write-Host "Copied: $item to $destinationDir"
    } else {
        Write-Host "Item not found: $item"
    }
}

Write-Host "Copy operation completed."
