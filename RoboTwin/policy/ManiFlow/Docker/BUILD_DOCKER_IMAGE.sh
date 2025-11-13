#!/bin/bash
# Example usage: bash BUILD_DOCKER_IMAGE.sh

# Default values
IMAGE_NAME="maniflow_robotwin2_docker" # change to your desired image name
TAG="latest"

# Print usage
usage() {
    echo "Usage: $0 [-n image_name] [-t tag]"
    echo "  -n: Image name (default: maniflow_robotwin_docker)"
    echo "  -t: Tag name (default: latest)"
    exit 1
}

# Parse command line arguments
while getopts "n:t:h" opt; do
    case ${opt} in
        n )
            IMAGE_NAME=$OPTARG
            ;;
        t )
            TAG=$OPTARG
            ;;
        h )
            usage
            ;;
        \? )
            usage
            ;;
    esac
done

# Full image name
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"

# Build the image
docker build \
    --network=host \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%d')" \
    -t "${FULL_IMAGE_NAME}" \
    .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Image built as: ${FULL_IMAGE_NAME}"
    echo ""
    echo "To run the container, use:"
    echo "docker run -it --rm ${FULL_IMAGE_NAME}"
else
    echo "Build failed!"
    exit 1
fi