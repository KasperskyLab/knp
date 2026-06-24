#!/bin/bash

DOCKER_COMMAND="docker"
IMAGE_NAME="kasperskydh/knp-sdk-image:latest"
ADD_DNS="--dns 8.8.8.8"
KNP_PACKAGES_VERSION="2.0.0_amd64"


function die()
{
    echo "$1" >&2
    exit "${2:-1}"
}


if ! command -v "${DOCKER_COMMAND}" &>/dev/null; then
    die "Docker isn't installed, you need to install docker package."
fi

echo "Pulling Docker image ${IMAGE_NAME}..."
"${DOCKER_COMMAND}" pull "${IMAGE_NAME}" || die "Image pull incomplete."

TMP_FILE=$(mktemp -u)
trap "rm -f \"${TMP_FILE}\"" EXIT

echo "Installing CUDA and GPU backend into the Docker image ${IMAGE_NAME}..."
"${DOCKER_COMMAND}" run --cidfile="${TMP_FILE}" ${ADD_DNS} -ti "${IMAGE_NAME}" \
  dpkg -i /knp/knp-gpu-cuda-backend_${KNP_PACKAGES_VERSION}.deb /knp/knp-gpu-cuda-backend-dev_${KNP_PACKAGES_VERSION}.deb \
  || die "Installation incomplete."

#echo "Removing old SDK image..."
#"${DOCKER_COMMAND}" image rm -f "${IMAGE_NAME}" || die "Cannot remove Docker image."

echo "Commiting changes into image..."
"${DOCKER_COMMAND}" commit -m "CUDA and KNP GPU backend installed" $(cat "${TMP_FILE}") "${IMAGE_NAME}"\
  || die "Cannot commit changes into Docker image."

echo "SDK image was installed successfully."
echo "You can run shell in the image, using following command: 'docker run -ti --rm ${IMAGE_NAME} bash'"
