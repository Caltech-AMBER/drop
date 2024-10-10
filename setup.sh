#!/bin/bash

dev=false

while [ $# -gt 0 ]; do
    case "$1" in
        --dev)
            dev=true
            shift  # Allows recommended system-level changes
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# set the CUBE_ROTATION_OBELISK_ROOT environment variable
export CUBE_ROTATION_OBELISK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create .env file for docker build args
env_file="$CUBE_ROTATION_OBELISK_ROOT/docker/.env"
if [ -f $env_file ]; then
    rm $env_file
fi
touch $env_file
echo "USER=$USER" > $env_file
echo "UID=$(id -u)" >> $env_file
echo "GID=$(id -g)" >> $env_file
echo "CUBE_ROTATION_OBELISK_ROOT=$CUBE_ROTATION_OBELISK_ROOT" >> $env_file
echo "DEV=$dev" >> $env_file
echo -e "\033[1;32m.env file populated under $CUBE_ROTATION_OBELISK_ROOT/docker!\033[0m"

# copy .cro_aliases to the docker directory
cp $CUBE_ROTATION_OBELISK_ROOT/scripts/.cro_aliases $CUBE_ROTATION_OBELISK_ROOT/docker/.cro_aliases
echo -e "\033[1;32m.cro_aliases copied to $CUBE_ROTATION_OBELISK_ROOT/docker!\033[0m"

# copy the cro directory to the docker directory
cp $CUBE_ROTATION_OBELISK_ROOT/README.md $CUBE_ROTATION_OBELISK_ROOT/docker/README.md
cp $CUBE_ROTATION_OBELISK_ROOT/pyproject.toml $CUBE_ROTATION_OBELISK_ROOT/docker/pyproject.toml
cp -r $CUBE_ROTATION_OBELISK_ROOT/cro $CUBE_ROTATION_OBELISK_ROOT/docker/cro

# if using the dev flag, create a persistent named docker volume for the cro folder if docker is installed
if command -v docker > /dev/null 2>&1 && [ "$dev" = true ] && [ ! "$(docker volume ls -q -f name=cro)" ]; then
    echo -e "\033[1;32mCreating persistent named volume for CRO data!\033[0m"
    docker volume create cro
    curr_dir=$(pwd)
    cd $CUBE_ROTATION_OBELISK_ROOT && \
        docker compose -f docker/docker-compose.yml run --build volume_instantiation && \
        docker compose -f docker/docker-compose.yml down volume_instantiation
    cd $curr_dir
fi
