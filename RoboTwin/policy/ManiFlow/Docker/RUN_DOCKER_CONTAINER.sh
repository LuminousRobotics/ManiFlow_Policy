#!/bin/bash

IMAGE_NAME="maniflow_robotwin2_docker" # change to your desired image name
TAG="latest"
CONTAINER_NAME="maniflow_robotwin2_container" # change to your desired container name, e.g., "robotwin_container"
LOCAL_PATH="/home/geyan/repos/dev/ManiFlow_Policy/" # change to your own repo path

# 检查是否已存在同名容器
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Found existing container named ${CONTAINER_NAME}"
    
    # 检查容器是否正在运行
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo "Container is running. Stopping it..."
        docker stop ${CONTAINER_NAME}
    fi
    
    echo "Removing existing container..."
    docker rm ${CONTAINER_NAME}
fi

# 运行新容器
echo "Starting new container..."
docker run -i \
    --name ${CONTAINER_NAME} \
    -v ${LOCAL_PATH}:${LOCAL_PATH} \
    -w ${LOCAL_PATH} \
    --gpus all \
    --ipc=host \
    -d \
    ${IMAGE_NAME}:${TAG}

# 等待容器完全启动
sleep 2

# 复制初始化脚本到容器
echo "Copying initialization script..."
docker cp initialize-docker-container.sh ${CONTAINER_NAME}:/workspace/

chmod +x initialize-docker-container.sh

# 执行初始化脚本
echo "Running initialization script..."
docker exec ${CONTAINER_NAME} bash /workspace/initialize-docker-container.sh

echo "Container is ready! To enter the container, use:"
echo "docker exec -it ${CONTAINER_NAME} bash"