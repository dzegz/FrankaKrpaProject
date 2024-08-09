# Prerequisites

Download and install python3 bindings for [echolib](https://github.com/vicoslab/echolib) into new virtualenv. Then set and export VIRTENV_ECHOLIB ponting to venv folder, or modify VIRTENV_ECHOLIB in variable in `start_camera_*.sh`.

Also create base echolib image for ubuntu 20.04 with
`docker build . -t echolib:20.04 --build-arg UBUNTU_VERSION=20.04`.
# 1. Download models

Download wanted model from the server:
-   depth_large -> `/storage/group/RTFM/household_set_v1+v2+v3+folded/3dof/backbone=tu-convnext_large_resize_factor=1_size=768x768/num_train_epoch=50/depth=True_hardneg=0/multitask_weight=uw_w_orient=1/weakly_supervised=True`
-   depth_base -> `/storage/group/RTFM/household_set_v1+v2+folded/3dof/backbone=tu-convnext_base_resize_factor=1_size=768x768/num_train_epoch=50/depth=True_hardneg=0/multitask_weight=uw_w_orient=1/weakly_supervised=True`
-   no_depth_large -> `/storage/group/RTFM/household_set_v1+v2+v3+folded/3dof/backbone=tu-convnext_large_resize_factor=1_size=768x768/num_train_epoch=50/depth=False_hardneg=0/multitask_weight=uw_w_orient=1/weakly_supervised=True`
-   no_depth_base -> `/storage/group/RTFM/household_set_v1+v2/backbone=tu-convnext_base_resize_factor=1_size=768x768/num_train_epoch=50/depth=False/multitask_weight=uw_w_orient=1/weakly_supervised=True`

Save the model in detector subfolder as `detector/{model_version}.pth`.

Download center_model from
`/storage/group/DIVID/pretrained-center-models/learnable-center-dilated-kernels-w_fg_cent=100_nn6.7.8.1_k=3_ch=16-loss=l1_lr=1e-4_hard_negs=4_on_occluded_augm_head_only_fully_syn=5k_no_edge_with_replacement`

Save it as `detector/center_model.pth`.

# 2. Build images
Modify variables inside `build_dockers.sh`. Run the script to build images.

# 3. Run the camera
We used Kinect V2 for the competation, but have now upgraded to Kinect Azure. Both are supported and build by `build_dockers.sh`, but by default it will start azure version. 

Make sure that Kinect is connected to the system and then run `start_camera_kinect_azure.sh`. This also starts echolib. You may need to modify the path to the python3 environment with echolib installed.

### Running on Kinect V2

To run on Kinect V2 you need to modify `detector/configs/*.py` files BEFORE (!) building docker images with `build_dockers.sh`:
  * in `detector/configs/*.py`: change to `CAMERA = "kinect"` instead of "kinect_azure"
Then use `start_camera_kinect_v2.sh` to start the camera.

# 4. Run the main program
Run `run_main.sh`. If you wish to use another model, without rebuilding the image, you can set the `MODEL_PATH` and `CONFIG_PATH` inside the script. This will mount the model and config inside the container.

This will also create 2 folders:
-   `competition_output`
-   `dataset`

When `space` is pressed on the cv2 display window, the program will save 4 files to `competition_output`:
-   Original image without any markers,
-   image with marked detected cloth corners and directions,
-   csv file with corner coordinates,
-   csv file with direction vectors.

When `s` is pressed on the cv2 display window, the program will save 2 files to `dataset`:
-   Current RGB image in `png` format,
-   current depth information as a `numpy` array.