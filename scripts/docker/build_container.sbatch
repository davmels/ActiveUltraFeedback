#!/bin/bash
#SBATCH --account=${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --container-writable
#SBATCH --job-name=build_container
#SBATCH --output=./logs/docker/build_%j.out

echo "================= Build Script ================="
cat scripts/docker/build_container.sh
echo -e "\n"

echo "================= Dockerfile ================="
cat Dockerfile
echo -e "\n"

echo "================= Building Docker image ================="
# Build the container using Podman
podman build -t activeuf:latest .

# Remove old (test) container if it exists
rm -f ${CONTAINERS_DIR:-/path/to/containers}/activeuf.sqsh

# Convert the Podman image to a SquashFS image
enroot import -o ${CONTAINERS_DIR:-/path/to/containers}/activeuf.sqsh podman://activeuf:latest

# Set permissions for the sqsh file
cd ${CONTAINERS_DIR:-/path/to/containers}/
setfacl -b activeuf.sqsh
chmod +r activeuf.sqsh

# Test container
echo "================= Testing container with one GPU ================="
srun --environment=activeuf python -m scripts.docker.test_single_gpu

echo "================= Testing container with multiple GPUs ================="
srun --environment=activeuf python -m scripts.docker.test_multi_gpu