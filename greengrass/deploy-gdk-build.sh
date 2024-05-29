#!/bin/bash
set -e

AWS_ACCOUNT_NUM=$AWS_ACCOUNT_NUM
AWS_REGION=$AWS_REGION
DEV_IOT_THING=$DEV_IOT_THING
DEV_IOT_THING_GROUP=$DEV_IOT_THING_GROUP

echo "Using AWS Account $AWS_ACCOUNT_NUM in Region $AWS_REGION ..."
echo "Deploying for IoT Thing $DEV_IOT_THING in IoT Group $DEV_IOT_THING_GROUP ..."

echo "Select update option:"
echo "1. Update Component 1: EdgeGenAI"
echo "2. Update Component 2: Jetson Docker"
echo "3. Update All components (EdgeGenAI & Jetson Docker)"
read -p "Enter your choice (1/2/3): " choice

# Setup IoT device
THING_ARN="arn:aws:iot:${AWS_REGION}:${AWS_ACCOUNT_NUM}:thing/${DEV_IOT_THING}"

jq -r --arg THING_ARN "$THING_ARN" '.targetArn=$THING_ARN' deployment-config.json > deployment-config.json.bak
mv deployment-config.json.bak deployment-config.json
jq -r --arg DEV_IOT_THING_GROUP "$DEV_IOT_THING_GROUP" '.deploymentName=$DEV_IOT_THING_GROUP' deployment-config.json > deployment-config.json.bak
mv deployment-config.json.bak deployment-config.json

## Based on choices, update components
### Update Component 1: EdgeGenAI
if [ $choice -eq 1 ] || [ $choice -eq 3 ]; then
    ## Build and Publish Component 1: EdgeGenAI
    # Automate the revision updates for GDK Build and Publish
    echo "Building and Publishing Component 1: EdgeGenAI ..."
    pushd components/com.aws.edge.genai
    if [ ! -f revision ]
    then
        echo 1 > revision
    fi
    REVISION_VER=$(cat revision)
    NEXT_REV=$((REVISION_VER+1))
    echo $NEXT_REV > revision
    echo Revision Version: $REVISION_VER
    popd
    # Set up greengrass component version
    export VERSION=$(cat version)
    COMPLETE_VER="$VERSION.$REVISION_VER"
    VER=${COMPLETE_VER}
    jq -r --arg VER "$VER" '.component[].version=$VER' components/com.aws.edge.genai/gdk-config.json > components/com.aws.edge.genai/gdk-config.json.bak
    mv components/com.aws.edge.genai/gdk-config.json.bak components/com.aws.edge.genai/gdk-config.json
    jq -r --arg AWS_REGION "$AWS_REGION" '.component[].publish.region=$AWS_REGION' components/com.aws.edge.genai/gdk-config.json > components/com.aws.edge.genai/gdk-config.json.bak
    mv components/com.aws.edge.genai/gdk-config.json.bak components/com.aws.edge.genai/gdk-config.json
    # GDK Build and GDK Publish
    pushd components/com.aws.edge.genai
    echo Removing old files if any
    rm -rf zip-build greengrass-build
    echo Building GDK component
    gdk component build
    echo Publishing GDK component
    gdk component publish
    popd
    jq --arg VERSION "$VER" '.components["com.aws.edge.genai"].componentVersion = $VERSION' deployment-config.json > deployment-config.json.bak
    mv deployment-config.json.bak deployment-config.json
    COMPONENTCONFIGURATION=$(jq -r '.ComponentConfiguration.DefaultConfiguration' components/com.aws.edge.genai/recipe.json)
    jq -r --arg COMPONENTCONFIGURATION "$COMPONENTCONFIGURATION" '.components["com.aws.edge.genai"].configurationUpdate.merge=$COMPONENTCONFIGURATION' deployment-config.json > deployment-config.json.bak
    mv deployment-config.json.bak deployment-config.json
    ####
fi

### Update Component 2: Jetson Docker
if [ $choice -eq 2 ] || [ $choice -eq 3 ]; then
    ## Build and Publish Component 2: Jetson Docker
    # Automate the revision updates for GDK Build and Publish
    echo "Building and Publishing Component 2: Jetson Docker ..."
    pushd components/com.aws.jetson.docker
    if [ ! -f revision ]
    then
        echo 1 > revision
    fi
    REVISION_VER=$(cat revision)
    NEXT_REV=$((REVISION_VER+1))
    echo $NEXT_REV > revision
    echo Revision Version: $REVISION_VER
    popd
    # Set up greengrass component version
    export VERSION=$(cat version)
    COMPLETE_VER="$VERSION.$REVISION_VER"
    VER=${COMPLETE_VER}
    jq -r --arg VER "$VER" '.component[].version=$VER' components/com.aws.jetson.docker/gdk-config.json > components/com.aws.jetson.docker/gdk-config.json.bak
    mv components/com.aws.jetson.docker/gdk-config.json.bak components/com.aws.jetson.docker/gdk-config.json
    jq -r --arg AWS_REGION "$AWS_REGION" '.component[].publish.region=$AWS_REGION' components/com.aws.jetson.docker/gdk-config.json > components/com.aws.jetson.docker/gdk-config.json.bak
    mv components/com.aws.jetson.docker/gdk-config.json.bak components/com.aws.jetson.docker/gdk-config.json
    DOCKER_IMAGE="${AWS_ACCOUNT_NUM}.dkr.ecr.${AWS_REGION}.amazonaws.com/genai-jetson:latest"
    DOCKER_COMMAND="docker run --runtime nvidia --rm --network host -p 65432:65432 ${DOCKER_IMAGE} python3 /opt/genai-jetson/main.py"
    DOCKER_SHUTDOWN_COMMAND="docker ps -a | grep ${DOCKER_IMAGE} | awk '{print $1}' | xargs -r docker stop && docker ps -a | grep ${DOCKER_IMAGE} | awk '{print $1}' | xargs -r docker rm"
    jq --arg DOCKER_COMMAND "$DOCKER_COMMAND" '.Manifests[].Lifecycle.Run.Script = $DOCKER_COMMAND' components/com.aws.jetson.docker/recipe.json > components/com.aws.jetson.docker/recipe.json.bak
    mv components/com.aws.jetson.docker/recipe.json.bak components/com.aws.jetson.docker/recipe.json
    DOCKER_IMAGE="docker:${DOCKER_IMAGE}"
    jq --arg DOCKER_IMAGE "$DOCKER_IMAGE" '.Manifests[].Artifacts[].URI = $DOCKER_IMAGE' components/com.aws.jetson.docker/recipe.json > components/com.aws.jetson.docker/recipe.json.bak
    mv components/com.aws.jetson.docker/recipe.json.bak components/com.aws.jetson.docker/recipe.json
    jq --arg DOCKER_SHUTDOWN_COMMAND "$DOCKER_SHUTDOWN_COMMAND" '.Manifests[].Lifecycle.Shutdown.Script = $DOCKER_SHUTDOWN_COMMAND' components/com.aws.jetson.docker/recipe.json > components/com.aws.jetson.docker/recipe.json.bak
    mv components/com.aws.jetson.docker/recipe.json.bak components/com.aws.jetson.docker/recipe.json
    # GDK Build and GDK Publish
    pushd components/com.aws.jetson.docker
    echo Removing old files if any
    rm -rf zip-build greengrass-build
    echo Building GDK component
    gdk component build
    echo Publishing GDK component
    gdk component publish
    popd
    jq --arg VERSION "$VER" '.components["com.aws.jetson.docker"].componentVersion = $VERSION' deployment-config.json > deployment-config.json.bak
    mv deployment-config.json.bak deployment-config.json
    ####
fi


# GDK Component Deployment
CONFIG_FILE="deployment-config.json"
RES=`aws greengrassv2 create-deployment --target-arn $THING_ARN --cli-input-json fileb://$CONFIG_FILE --region $AWS_REGION`
echo Greengrass Deployment ID: ${RES}
