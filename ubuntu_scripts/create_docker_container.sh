#!/bin/bash

# Function to check if a port is available
is_port_available() {
  if command -v ss >/dev/null 2>&1; then
    # Use ss if available
    ! ss -tnl | grep -q ":$1 "
  elif command -v netstat >/dev/null 2>&1; then
    # Fallback to netstat if ss is not available
    ! netstat -tnl | grep -q ":$1 "
  else
    echo "Error: could not find 'ss' or 'netstat' to check port availability."
    exit 1
  fi
}

# Function to display the process using a specific port
display_process_using_port() {
  if command -v netstat >/dev/null 2>&1; then
    # Use netstat if available
    echo "Warning: Port $1 is occupied by:"
    netstat -tnlp | grep ":$1 "
  elif command -v ss >/dev/null 2>&1; then
    # Fallback to ss if netstat is not available
    ss -tnlp | grep ":$1 "
  else
    echo "Error: could not find 'netstat' or 'ss' to display the process using the port."
    exit 1
  fi
}

# Generate a random port in the range 20000-65000 and check if it's available
get_random_port() {
  while :; do
    local port=$(shuf -i 20000-65000 -n 1)
    if is_port_available "$port"; then
      echo $port
      break
    else
      display_process_using_port "$port"
    fi
  done
}

#print help
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "Usage: $0 <docker_image_name> <container_name> <number_of_ports> <ssh_port>"
  echo "docker_image_name: The name of the Docker image to run."
  echo "container_name: The name of the Docker container to create."
  echo "number_of_ports: The number of ports to map from the container to the host."
  echo "ssh_port: The port to map for SSH. If not provided, a random port will be used."
  echo "Example: $0 my_docker_image my_container 3 2222"
  exit 1
fi

# Check if an image name, container name, and number of ports are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <docker_image_name> <container_name> <number_of_ports>"
  exit 1
fi

IMAGE_NAME="$1"
CONTAINER_NAME="$2"
NUMBER_OF_PORTS="$3"

# Generate random host ports and create port mappings
PORT_MAPPINGS=()
if [ -n "$4" ]; then
  if ! is_port_available $4; then
    display_process_using_port $4
    echo "Using random port instead"
    PORT_START=$(get_random_port) # Get a random available host port
  else
    echo "Using specific port $4"
    PORT_START=$4
  fi
else
  echo "Using random port"
  PORT_START=$(get_random_port) # Get a random available host port
fi

# SSH port mapping
PORT_MAPPINGS="-p $PORT_START:22"

i=1
while [ $i -lt $((NUMBER_OF_PORTS + 1)) ]; do
  PORT=$((PORT_START + i))
  if ! is_port_available "$PORT"; then
    PORT=$(get_random_port) # Find a new available port
  fi
  PORT_MAPPINGS="$PORT_MAPPINGS -p $PORT:$PORT"
  i=$((i + 1))
done
PORTS="$PORT_MAPPINGS"

# Show the full Docker run command to the user
DOCKER_COMMAND="docker run -it --gpus all --privileged -d -u 0 \
    -v /data1/hanhu2/python/interesting/:/code \
    -v /data1/hanhu2/data/:/data \
    -v ~/.ssh:/host_ssh \
    --name \"$CONTAINER_NAME\" \
    --shm-size 32G \
    $PORTS \
    \"$IMAGE_NAME\" /bin/bash"

echo "You are about to execute the following command:"
echo $DOCKER_COMMAND

# Ask for user confirmation
read -p "Do you want to proceed with this command? [y/n]: " CONFIRM
case $CONFIRM in
[yY])
  eval $DOCKER_COMMAND
  echo "Container \"$CONTAINER_NAME\" is starting."
  ;;
[nN])
  echo "Container creation aborted."
  exit 0
  ;;
*)
  echo "Invalid input. Please answer y or n."
  exit 1
  ;;
esac

echo "Creating container $CONTAINER_NAME is done."
read -p "Do you want to enable ssh in the container? [y/n]: " CONFIRM
case $CONFIRM in
[yY])
    echo "Start to enable ssh in the container."
    # install ssh in the container
    docker exec -it $CONTAINER_NAME apt-get update
    docker exec -it $CONTAINER_NAME apt-get install -y openssh-server
    docker exec -it $CONTAINER_NAME mkdir /var/run/sshd
    docker exec -it $CONTAINER_NAME /usr/sbin/sshd

    # modify root password
    for i in {1..3}
    do 
        read -p "Please input the password for root: " ROOT_PASSWORD
        read -p "Please confirm the password for root: " ROOT_PASSWORD_CONFIRM
        if [ "$ROOT_PASSWORD" == "$ROOT_PASSWORD_CONFIRM" ]; then
            break
        else
            echo "The passwords do not match. Please try again."
        fi
    done

    if [ "$i" -eq 3 ]; then
        echo "You have tried 3 times. Please restart the script."
        exit 1
    fi

    docker exec -it $CONTAINER_NAME bash -c "echo 'root:$ROOT_PASSWORD' | chpasswd"

    # delete "PermitRootLogin" exists, add it
    docker exec -it $CONTAINER_NAME sed -i '/PermitRootLogin yes/d' /etc/ssh/sshd_config
    docker exec -it $CONTAINER_NAME sh -c 'echo "PermitRootLogin yes" >> /etc/ssh/sshd_config'

    docker exec -it $CONTAINER_NAME sed -i 's/#PasswordAuthentication yes/d' /etc/ssh/sshd_config
    docker exec -it $CONTAINER_NAME sh -c 'echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config'

    docker exec -it $CONTAINER_NAME /etc/init.d/ssh restart

    echo "SSH is enabled in the container $CONTAINER_NAME.\n
        port: $PORT_START, user: root, password: $ROOT_PASSWORD"
    ;;
[nN])
    echo "SSH is not enabled in the container. container $CONTAINER_NAME is ready to use."
    ;;
*)
    echo "Invalid input. Please answer y or n."
    exit 1
    ;;
esac
