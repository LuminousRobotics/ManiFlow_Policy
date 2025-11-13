# Build Docker Container for ManiFlow in RoboTwin

1. Prerequisites

   - Docker installed on your system

2. Build the Docker image

   ```bash
   cd RoboTwin/policy/ManiFlow/Docker
   bash BUILD_DOCKER_IMAGE.sh
   ```

3. Configure the container (only run once)
   ```bash
   # Update $LOCAL_PATH in these scripts to your repo path
   # Edit RUN_DOCKER_CONTAINER.sh and initialize-docker-container.sh
   
   bash RUN_DOCKER_CONTAINER.sh
   ```

4. Run inside the container

   ```bash
   docker exec -it robotwin2_container bash # robotwin2_container is the container name defined in RUN_DOCKER_CONTAINER.sh
   cd YOUR_PATH_TO_RoboTwin  # Should match $LOCAL_PATH from step 3
   ```
