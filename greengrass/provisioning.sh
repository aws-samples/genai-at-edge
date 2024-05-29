#!/bin/bash
set -e

# Function to get system details
function get_os_details()
{
    echo "Determining OS version"
    OS_ID=""
    OS_VERSION=""
    OS_NAME=""
    OS_ARCH=""

    # we only support linux version with /etc/os-release present
    if [ -r /etc/os-release ]; then
        # load the variables in os-release
        . /etc/os-release
        OS_ID=$ID
        OS_VERSION_ID=$VERSION_ID
        OS_NAME=$NAME
        OS_ARCH=`uname -m`
    fi

    if [ -z $OS_ID ]; then
        echo $OS_ID
        echo "Cannot recognize operating system. Exiting."
        exit
    fi

    if [ "$OS_ID" != "ubuntu" ]; then
        echo "Currently only works for Ubuntu. Exiting."
        exit
    fi

    echo "=== You are running installing on OS=$OS_NAME,ID=$OS_ID,VERSION=$OS_VERSION_ID,ARCH=$OS_ARCH ==="
}

# Function to install packages
function install_packages()
{
    sudo apt update -y
    sudo apt install -y git zip unzip build-essential wget curl libpng-dev autoconf libtool pkg-config libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev libpulse-dev software-properties-common coreutils java-common

    if ! [ -x "$(command -v aws)" ]; then
        echo "AWS is NOT installed. Trying to Install."
        if [ $OS_ARCH == "aarch64" ]; then
            curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
        elif [ $OS_ARCH == "x86_64" ]; then
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        else
            echo "ARM64 or x86_64 system is not identified. Exiting."
            exit
        fi
        unzip awscliv2.zip
        sudo ./aws/install
    else
        echo "AWS is installed."
    fi

    echo "Configuring AWS."
    if [ $? -eq 0 ]; then
        aws configure 2>&1 || echo "AWS Configure failed."
    fi

    echo "Installing Greengrass Dependencies"
    if [ $OS_ARCH == "aarch64" ]; then
        wget https://corretto.aws/downloads/latest/amazon-corretto-11-aarch64-linux-jdk.deb
        sudo dpkg --install amazon-corretto-11-aarch64-linux-jdk.deb
        rm -rf amazon-corretto-11-aarch64-linux-jdk.deb
    elif [ $OS_ARCH == "x86_64" ]; then
        sudo apt install default-jdk
    else
        echo "ARM64 or X86_64 system is not identified. Exiting."
        exit
    fi
    sudo useradd --system --create-home ggc_user
    sudo groupadd --system ggc_group
}

# Function to provision IoT device
function provision_device()
{
    echo "Provisioning Device for IoT Greengrass"
    echo -n "Enter IoT Thing Name [default=GreengrassThing]: "
    read IOT_THING_NAME
    IOT_THING_NAME=${IOT_THING_NAME//[[:blank:]]/}
    if [ -z "$IOT_THING_NAME" ];  then
        IOT_THING_NAME="GreengrassThing"
    fi
    echo "IoT Thing Name entered is: $IOT_THING_NAME"
    echo -n "Enter IoT Thing Group Name [default=GreengrassThingGroup]: "
    read IOT_THING_GROUP
    IOT_THING_GROUP=${IOT_THING_GROUP//[[:blank:]]/}
    if [ -z "$IOT_THING_GROUP" ];  then
        IOT_THING_GROUP="GreengrassThingGroup"
    fi
    echo "IoT Thing Group entered is: $IOT_THING_GROUP"
    echo "Provisioning the device **$IOT_THING_NAME** within the group **$IOT_THING_GROUP**"

    cd
    mkdir -p greengrass_files
    pushd greengrass_files
    curl -s https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip > greengrass-nucleus-latest.zip
    unzip -o greengrass-nucleus-latest.zip -d GreengrassInstaller && rm greengrass-nucleus-latest.zip
    sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE -jar ./GreengrassInstaller/lib/Greengrass.jar --aws-region us-east-1 --thing-name $IOT_THING_NAME --thing-group-name $IOT_THING_GROUP --component-default-user ggc_user:ggc_group --provision true --setup-system-service true --deploy-dev-tools true --thing-policy-name GreengrassV2IoTThingPolicy --tes-role-name GreengrassV2TokenExchangeRole
    popd
    rm -rf greengrass_files
}

# Function to install dependencies
function install_dependencies()
{
    echo "Installing dependencies"
    sudo apt-get install docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker ggc_user
    sudo usermod -aG video ggc_user
}

# Main Function
function main()
{
    get_os_details
    install_packages
    provision_device
    install_dependencies
}

main