#!/bin/bash

SCRIPT_PATH="$(dirname $(realpath "${BASH_SOURCE[0]}"))"

DOCKER_COMMAND="docker"

SDK_IMAGE_BASE_NAME="knp-sdk-image"
BUILD_IMAGE_BASE_NAME="knp-build-image"

SDK_IMAGE_NAME="kasperskydh/${SDK_IMAGE_BASE_NAME}:latest"
SDK_IMAGE_FILE="${SCRIPT_PATH}/${SDK_IMAGE_BASE_NAME}.txz"

EULA_FILE="${SCRIPT_PATH}/LICENSE.txt"

ADD_DNS="--dns 8.8.8.8"
KNP_PACKAGES_VERSION="2.0.0_amd64"


function die()
{
    echo "$1" >&2
    exit "${2:-1}"
}


function show_eula()
{
    while true; do
        more "$1"
		echo
        read -p "Do you accept the EULA? (y/n): " choice
        case "$choice" in
            [Yy]*) echo "EULA accepted.";
            break;;
            [Nn]*) die "EULA rejected. Exiting."
            ;;
            *) echo "Please answer yes (y) or no (n).";;
        esac
    done
}


if ! command -v "${DOCKER_COMMAND}" &>/dev/null; then
    die "Docker isn't installed, you need to install docker package."
fi

if [ -e "${EULA_FILE}" ]; then
    show_eula "${EULA_FILE}"
fi

if [ -e "${SDK_IMAGE_FILE}" ]; then
    echo "SDK image file \"${SDK_IMAGE_FILE}\" was found, loading..."
    "${DOCKER_COMMAND}" load -i "${SDK_IMAGE_FILE}" || die "Image loading error."
    echo "SDK image file loaded successfully."
else
    echo "Pulling Docker image ${SDK_IMAGE_NAME}..."
    "${DOCKER_COMMAND}" pull "${SDK_IMAGE_NAME}" || die "Image pull incomplete."
fi

TMP_FILE=$(mktemp -u)
trap "rm -f \"${TMP_FILE}\"" EXIT

echo "Installing CUDA and GPU backend into the Docker image ${SDK_IMAGE_NAME}..."
"${DOCKER_COMMAND}" run --cidfile="${TMP_FILE}" ${ADD_DNS} -ti "${SDK_IMAGE_NAME}" \
  dpkg -i /knp/knp-gpu-cuda-backend_${KNP_PACKAGES_VERSION}.deb /knp/knp-gpu-cuda-backend-dev_${KNP_PACKAGES_VERSION}.deb \
  || die "Installation incomplete."

#echo "Removing old SDK image..."
#"${DOCKER_COMMAND}" image rm -f "${IMAGE_NAME}" || die "Cannot remove Docker image."

echo "Commiting changes into image..."
"${DOCKER_COMMAND}" commit -m "CUDA and KNP GPU backend installed" $(cat "${TMP_FILE}") "${SDK_IMAGE_NAME}"\
  || die "Cannot commit changes into Docker image."

echo "SDK image was installed successfully."
echo "You can run shell in the image, using following command: 'docker run -ti --rm ${SDK_IMAGE_NAME} bash'"
