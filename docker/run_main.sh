# Set if you want to override config/model stored in the image
CONFIG_PATH=${1:-""}
MODEL_PATH=${2:-""}
CENTER_MODEL_PATH=${3:-""}

CONTAINER_ARGS=""

if [ ! -z "$CONFIG_PATH" ]; then
    CONTAINER_ARGS="$CONTAINER_ARGS -v $CONFIG_PATH:/opt/cedirnet-dev/src/config/model_args.py"
fi
if [ ! -z "$MODEL_PATH" ]; then
    CONTAINER_ARGS="$CONTAINER_ARGS -v $MODEL_PATH:/opt/model.pth"
fi
if [ ! -z "$CENTER_MODEL_PATH" ]; then
    CONTAINER_ARGS="$CONTAINER_ARGS -v $CENTER_MODEL_PATH:/opt/center_model.pth"
fi

DEBUG=${DEBUG:-0}
if [ "${DEBUG}" -eq 1 ]; then
    CONTAINER_ARGS="$CONTAINER_ARGS -v $(pwd)/..:/opt/cedirnet-dev/tools/competition/ --entrypoint=/bin/bash"
fi

echo Creating folders for outputs
mkdir -p dataset
mkdir -p competition_output

docker run -it --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/echo.sock:/tmp/echo.sock \
    -v $(pwd)/dataset:/opt/dataset \
    -v $(pwd)/competition_output:/opt/competition_output \
    -e DISPLAY="$DISPLAY" \
    --gpus all \
    $CONTAINER_ARGS \
    cloth-corner-detector


