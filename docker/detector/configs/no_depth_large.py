"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

import torchvision
if 'InterpolationMode' in dir(torchvision.transforms):
	from torchvision.transforms import InterpolationMode
else:
	from PIL import Image as InterpolationMode

import torch
from utils import transforms as my_transforms

# DATA_DIR=os.environ.get('DATA_DIR')
# DATA_DIR = '/storage/datasets/ClothDataset/household_set/'

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')
STORAGE_DIR=os.environ.get('STORAGE_DIR',default='/storage')

ENABLE_6DOF = False
NUM_FIELDS = 3 + (6 if ENABLE_6DOF else 2)

# DATASET_NAME = 'rtfm'
DATASET_NAME = 'household_set'
# WEAKLY_SUPERVISED = False
WEAKLY_SUPERVISED = True
SIZE = 768
# SIZE = 384
# SIZE = 256
BACKBONE='tu-convnext_large'
# BACKBONE='resnet101'

regression_region = 15
# regression_region = 30
orientation_weight = 1
# orientation_weight = 4
USE_DEPTH = False

# DATASET_TYPE = 'test_synthetic'
DATASET_TYPE = 'test_household'

#CAMERA = "kinect" # "allied"
CAMERA = "kinect_azure"


DATA_DIR = '/storage/datasets/ClothDataset/household_set/' if 'household' in DATASET_TYPE else '/storage/datasets/ClothDataset/mujoco/'

# model_dir = os.path.join(OUTPUT_DIR,'screws_m6x20','6dof' if ENABLE_6DOF else '',
model_dir = os.path.join(OUTPUT_DIR, 'rtfm','6dof' if ENABLE_6DOF else '3dof',
						  'backbone={args[model][kwargs][backbone]}',
						  'num_train_epoch={args[train_settings][n_epochs]}', f'weakly_supervised={"True" if WEAKLY_SUPERVISED else "False"}',
						  f'size={SIZE}x{SIZE}'+f',regression_region={regression_region}, orientation_weight={orientation_weight}'+f'{",depth" if USE_DEPTH else ""}')

# save_dir=os.path.join(OUTPUT_DIR, DATASET_NAME, dof_name,
# 						  'backbone={args[model][kwargs][backbone]}',
# 						  'num_train_epoch={args[n_epochs]}',f'weakly_supervised={"True" if WEAKLY_SUPERVISED else "False"}', f'size={SIZE}x{SIZE}'+f',regression_region={regression_region}, orientation_weight={orientation_weight}'),

args = dict(
    threshold=0.75,
	use_depth = USE_DEPTH,
	size=SIZE,
	camera=CAMERA,
	cuda=True,
	# display=False,
	display=True,
	# display_from_ap50=True,
	display_from_ap50=False,
	display_individual_predictions=False,
	display_individual_predictions_thr=0.01,
	autoadjust_figure_size=True,
	# display to file only
	display_to_file_only=True,
	# display_to_file_only=False,
	# autoadjust_figure_size=False,
	groundtruth_loading = True,

	# generate rotated bbox from '3sigma' or from 'minmax' or from opencv
	generate_rot_bbox_from='3sigma', # '3sigma' or 'minmax'

	# save=False,
	save=True,
	# eval=True,
	save_dir=os.path.join(model_dir,'{args[dataset][kwargs][type]}_results{args[eval_epoch]}','GTCenters'),
	checkpoint_path=os.path.join(model_dir,'checkpoint{args[eval_epoch]}.pth'),

	center_checkpoint_path=os.path.join(STORAGE_DIR,'group/DIVID/pretrained-center-models/',
										'learnable-center-dilated-kernels-w_fg_cent=100_nn6.7.8.1_k=3_ch=16-loss=l1_lr=1e-4_hard_negs=4_on_occluded_augm_head_only_fully_syn=5k_no_edge_with_replacement',
										'checkpoint.pth'),

	# eval_epoch='_050',
	eval_epoch='',
	apply_grad_norm=False, # 'simple', 'per-error-type', or None/False
	pretrained_center_name='pretrained-learnable-center=nn6.7.8.1_no_finetuning',

	eval=dict(
		# available score types ['mask', 'center', 'hough_energy', 'edge_to_area_ratio_of_mask', 'avg(mask_pix)', 'avg(hough_pix)', 'avg(projected_dist_pix)']
		score_combination_and_thr=[
			{
			#'center': [0.1,0.01,0.05,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.94,0.99],
			 'center': [0.5],
			 'edge_to_area_ratio_of_mask':None, 'avg(mask_pix)':None},
		],
		score_thr_final=[0.7],
		skip_center_eval=True,
		orientation=dict(
			display_best_threshold=False,
			# tau_thr=[5,20] # 5 == used in paper "Locating Objects Without Bounding Boxes" by Ribera et al.
			tau_thr=[20] # 5 == used in paper "Locating Objects Without Bounding Boxes" by Ribera et al.
		),
	),
	visualizer=dict(name='OrientationVisualizeTest',
					# opts=dict(show_rot_axis=(False,False,True))),
					# opts=dict(show_rot_axis=(True,True,True))),
					opts=dict(show_rot_axis=(True,))),


	dataset={
		'name': DATASET_NAME,
		'kwargs': {
			'use_depth': USE_DEPTH,
			'normalize': False,
			'root_dir': DATA_DIR,
			'fixed_bbox_size': regression_region,
			'type': DATASET_TYPE,
			'output_single_orientation_only':(False if ENABLE_6DOF else 'x') ,
			'transform': my_transforms.get_transform([
				
				{
					'name': 'ToTensor',
					'opts': {
						# 'keys': ('image', 'instance', 'label', 'ignore'),
						# 'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor),
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
						'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor) + ((torch.FloatTensor, ) if USE_DEPTH else ()),
					}
				},

				{
					'name': 'Resize',
					'opts': {
						# 'keys': ('image', 'instance', 'label', 'difficult'),
						# 'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
						'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.NEAREST) + ((InterpolationMode.BILINEAR, ) if USE_DEPTH else ()),
						'keys_bbox': ('center',),
						# 'size': (SIZE, SIZE),
						'size': (SIZE, 896),
					}
				},
				
			]),
			'MAX_NUM_CENTERS':2*1024,
		},
		'centerdir_gt_opts': dict(
			ignore_instance_mask_and_use_closest_center=True,
			# ignore_instance_mask_and_use_closest_center=False,
			center_ignore_px=3,
			# center_ignore_px=1,

			#polar_gt_cache=os.path.join(STORAGE_DIR,'local/ssd/cache-polar/mall/point_supervision_box=30px_train_patches_512x512'),
			extend_instance_mask_weights=False,

			#backbone_output_cache=os.path.join(STORAGE_DIR,'local/ssd/cache-polar/mall_output'),
			load_cached_backbone_output_probability=0,
			use_cached_backbone_output=False,
			save_cached_backbone_output_only=False,
			MAX_NUM_CENTERS=2*1024,
		),

		'batch_size': 1,
		'workers': 0,
	},

	model=dict(
		name='fpn',
		kwargs={
			# 'backbone': 'resnet50',
			"pretrained": False,
			'backbone': BACKBONE,
			'num_classes': [NUM_FIELDS, 1],
			'use_custom_fpn': True,
			'add_output_exp': False,
			'use_depth': USE_DEPTH,
			'fpn_args': {
				# 'decoder_dropout':0
				'decoder_segmentation_head_channels': 64,
				'classes_grouping': [(0, 1, 2, 5), (3, 4)],
				'depth_mean': 0.96, 'depth_std': 0.075,
			},
			'init_decoder_gain': 0.1
		},
	),
	center_model=dict(
		name='CenterOrientationEstimator',
		use_learnable_center_estimation=True,

		kwargs=dict(
			use_centerdir_radii = False,
			
			# use vector magnitude as mask instead of regressed mask
			use_magnitude_as_mask=True,
			# thresholds for conv2d processing
			local_max_thr=0.01,
			#local_max_thr=0.1,
			# local_max_thr=0.25,
			# local_max_thr=0.5,
			mask_thr=0.01, hough_thr=5000, local_max_thr_use_abs=True,
			# thresholds for mask suppression using hough estimation (hough_mask_thr == thr applied to magnitude)
			# suppression_by_mask=True,
			suppression_by_mask=False,
			hough_mask_thr=0.2,
			# number of jumps/hops for mask estimation from votes
			hough_num_hops=20,
			hough_hop_distance_factor=0.9,
			hough_hop_distnace_walked_ratio=0.1,
			### dilated neural net as head for center detection
			ignore_centerdir_magnitude=True,
			ignore_cls_prediction=True,
			use_dilated_nn=True,
			dilated_nn_args=dict(
				return_sigmoid=False,
				# single scale version
				# inner_ch=16,
				# inner_kernel=3,
				# dilations=[1, 4, 8, 16, 32, 48],
				# multiscale version
				# inner_ch=16,
				# inner_kernel=3,
				# dilations=[1, 4, 12],
				# min_downsample=32
				# single scale version (nn6)
				inner_ch=16,
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				use_centerdir_radii=False,
				use_centerdir_magnitude=False,
				use_cls_mask=False
				),
			augmentation_name='CenterAugmentator',
			augmentation_kwargs=dict(
				occlusion_probability=0,
				occlusion_type='circle',
				occlusion_distance_type='larger',  # 'fixed',
				occlusion_center_jitter_px=0,
				gaussian_noise_probability=0,
				gaussian_noise_blur_sigma=3,
				gaussian_noise_std_polar=[0.1, 2.0],
				gaussian_noise_std_mask=[0.1, 2.0]

			),
			scale_r=1.0,  # 1024
			scale_r_gt=1,  # 1
			use_log_r=False,
			use_log_r_base='10',
			enable_6dof=ENABLE_6DOF,
		),

	),
	num_vector_fields=NUM_FIELDS,

	# settings from train config needed for automated path construction
	train_settings=dict(
		train_dataset=dict(
			kwargs=dict(
				fixed_bbox_size=30,
				BORDER_MARGIN_FOR_CENTER=0,
				remove_out_of_bounds_centers=True,
				aug_random_center_pertub=0,
				resize_factor=1.0,
			),
			batch_size=32,
			hard_samples_size=0,
			centerdir_gt_opts=dict(center_ignore_px=3),
		),
		model=dict(
			lr=1e-4,
			weight_decay=0,
		),
		n_epochs=50,
		apply_grad_norm=False,
		loss_opts=dict(
			loss_weighted_by_distance_gauss=0,
			# sigma for converting R distance to weight (3 -> 0.1 at 40 pix distance)
			border_weight=1.0,
			border_weight_px=0
		),
		loss_w=dict(w_r=0),
	)
)

def get_args():
	return copy.deepcopy(args)
