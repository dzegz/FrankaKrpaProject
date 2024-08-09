#!/usr/bin/python
import os, time
import cv2
import csv
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

import numpy as np
import scipy

import torch
from torchvision import transforms
from datasets import get_centerdir_dataset
from models import get_model, get_center_model
from utils.utils import variable_len_collate
from kinect_stream_iterator import KinectStreamIterator
from kinect_azure_stream_iterator import KinectAzureStreamIterator
from allied_stream_iterator import AlliedStreamIterator

class Inference:
    def __init__(self, args):
        # if args['display'] and not args.get('display_to_file_only'):
        # if True:
        #     # plt.switch_backend('TkAgg')
        #     plt.ion()
        # else:
        #     plt.ioff()
        #     plt.switch_backend("agg")

        if args.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self.args = args
        self.show_regression_output = args.get('show_regression_output')
        self.crop = args.get('crop')
        

        # set device
        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")

    def initialize(self):
        args = self.args

        ###################################################################################################
        # set dataset and model
        self.dataset_it, self.model, self.center_model = self._construct_dataset_and_processing(args, self.device)


    #@classmethod
    def _construct_dataset_and_processing(self, args, device):

        ###################################################################################################
        # dataloader
        # dataset_workers = args['dataset']['workers'] if 'workers' in args['dataset'] else 0
        # dataset_batch = args['dataset']['batch_size'] if 'batch_size' in args['dataset'] else 1

        # from utils import transforms as my_transforms
        # args['dataset']['kwargs']['transform'] = my_transforms.get_transform([
        #     { 'name': 'Padding', 'opts': { 'keys': ('image',), 'pad_to_size_factor': 32 } },
        #     { 'name': 'ToTensor', 'opts': { 'keys': ('image',), 'type': (torch.FloatTensor) } },
        # ])

        # dataset, _ = get_centerdir_dataset(args['dataset']['name'], args['dataset']['kwargs'], no_groundtruth=True)

        # dataset_it = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch, shuffle=False, drop_last=False,
        #                                          num_workers=dataset_workers, pin_memory=True if args['cuda'] else False,
        #                                          collate_fn=variable_len_collate)

        if args["camera"] == "kinect":
            dataset_it = KinectStreamIterator()
        elif args["camera"] == "kinect_azure":
            dataset_it = KinectAzureStreamIterator()
        elif args["camera"] == "allied":
            dataset_it = AlliedStreamIterator()

            if args["use_depth"]:
                raise ValueError("Allied camera doesn't support depth!")
        else:
            raise ValueError("Please provide valid camera in config")

        ###################################################################################################
        # load model
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model.init_output(args['num_vector_fields'])
        model = torch.nn.DataParallel(model).to(device)

        # prepare center_model and center_estimator based on number of center_checkpoint_path that will need to be processed

        center_checkpoint_path = args.get('center_checkpoint_path')

        center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'],
                                        is_learnable=args['center_model'].get('use_learnable_center_estimation'),
                                        use_fast_estimator=True)

        center_model.init_output(args['num_vector_fields'])
        center_model = torch.nn.DataParallel(center_model).to(device)

        ###################################################################################################
        # load snapshot
        if os.path.exists(args['checkpoint_path']):
            print('Loading from "%s"' % args['checkpoint_path'])
            state = torch.load(args['checkpoint_path'])
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                center_model.load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

        if args['center_model'].get('use_learnable_center_estimation'):
            if os.path.exists(center_checkpoint_path):
                print('Loading center model from "%s"' % center_checkpoint_path)
                state = torch.load(center_checkpoint_path)
                if 'center_model_state_dict' in state:
                    if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                        checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                        center_input_weights = center_model.module.instance_center_estimator.conv_start[0].weight
                        if checkpoint_input_weights.shape != center_input_weights.shape:
                            state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]

                            print('WARNING: #####################################################################################################')
                            print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                            print('WARNING: #####################################################################################################')

                    center_model.load_state_dict(state['center_model_state_dict'], strict=False)
            else:
                raise Exception('checkpoint_path {} does not exist!'.format(center_checkpoint_path))

        return dataset_it, model, center_model

    #########################################################################################################
    ## MAIN RUN FUNCTION
    def run(self):
        args = self.args

        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((args["size"], args["size"]))
        threshold = args["threshold"]
        print(f"Using threshold: {threshold}")

        os.makedirs(os.path.join(args["data_save_path"], "rgb"), exist_ok=True)
        os.makedirs(os.path.join(args["data_save_path"], "depth"), exist_ok=True)
        save_index = len(os.listdir(os.path.join(args["data_save_path"], "rgb")))

        os.makedirs(args["competition_save_path"], exist_ok=True)
        competition_index = len(os.listdir(args["competition_save_path"])) // 4

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Output", 1280,720)

        if self.show_regression_output:
            cv2.namedWindow("Vector regression", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Vector regression", 1280,720)

            cv2.namedWindow("Center regression", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Center regression", 1280,720)

        time_array = dict(model=[],center=[],post=[])
        with torch.no_grad():
            model = self.model
            center_model = self.center_model
            dataset_it = self.dataset_it

            model.eval()
            center_model.eval()

            for sample in dataset_it:
                torch.cuda.synchronize()

                raw_input = None
                if args["use_depth"]:
                    depth = sample["depth"].astype(np.float32)

                    invalid_mask = np.isinf(depth) | np.isnan(depth) | (depth > 1e4) | (depth<0)
                    depth[invalid_mask] = depth[~invalid_mask].mean()
                    depth /= np.max(depth)

                    raw_input = torch.cat((to_tensor(sample["image"]), 
                                           to_tensor(depth)))
                    model_input = torch.cat((resize(to_tensor(sample["image"])), 
                                             resize(to_tensor(depth))))
                                             
                else:
                    raw_input = to_tensor(sample["image"])
                    model_input = resize(to_tensor(sample["image"]))

                # run main model
                output_batch_ = model(model_input)

                torch.cuda.synchronize()

                # run center detection model
                center_output = center_model(output_batch_)
                output, center_pred, center_heatmap, pred_mask, pred_angle = [center_output[k] for k in ['output','center_pred','center_heatmap','pred_mask', 'pred_angle']]

                # make sure data is copied
                torch.cuda.synchronize()
                end = time.time()

                #display_image = model_input.permute(1, 2, 0).numpy()
                display_image = sample["image"]
                
                height = display_image.shape[0]
                width = display_image.shape[1]
                
                cropped_image = display_image
                top_crop = 0
                left_crop = 0
                if self.crop:
                    #top_crop = 120
                    #side_crop = int((width - (height - top_crop) / 3 * 4) / 2)
                    #cropped_image = display_image[top_crop:, side_crop:width-side_crop]

                    top_crop = int(self.crop['top'] if 'top' in self.crop else 0)
                    left_crop = int(self.crop['left'] if 'left' in self.crop else 0)
                    bottom_crop = int(height - (self.crop['bottom'] if 'bottom' in self.crop else 0))
                    right_crop = int(width - (self.crop['right'] if 'right' in self.crop else 0))
                    
                    display_image = display_image[top_crop:bottom_crop, left_crop:right_crop]
                    cropped_image = display_image[top_crop:bottom_crop, left_crop:right_crop]

                display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

                predictions = center_pred[0][center_pred[0,:,0] == 1][:,1:].cpu().numpy()
                idx = np.argsort(predictions[:, -1])
                idx = idx[::-1]
                predictions = predictions[idx, :]
                angles = pred_angle[0].cpu().numpy()[idx, :]

                points = []
                vectors = []
                
                for prediction, angle in zip(predictions, angles):
                    if prediction[3] < threshold:
                        continue

                    scale_y = sample["image"].shape[0] / args["size"]
                    scale_x = sample["image"].shape[1] / args["size"]

                    x1, y1 = int(prediction[0] * scale_x) * 2, int(prediction[1] * scale_y) * 2

                    x1 -= left_crop
                    y1 -= top_crop

                    if x1 < 0 or display_image.shape[1] < x1:
                        continue

                    if y1 < 0 or display_image.shape[0] < y1:
                        continue

                    angle = (angle - 180) * np.pi / 180
                    length = 50
                    x2 = int(x1 + length * np.cos(angle))
                    y2 = int(y1 + length * np.sin(angle))

                    display_image = cv2.line(display_image, (x1, y1), (x2, y2), color=(255,0,0), thickness=4) 
                    display_image = cv2.circle(display_image, (x1, y1), radius=8, color=(0,0,255), thickness=-1)

                    points.append((x1, y1))
                    vectors.append((x2, y2))

                cv2.imshow("Output", cv2.flip(display_image, 1))
                
                if self.show_regression_output:
                    def apply_colormap(im, norm=None):
                        if norm is None:
                            im = (((im-im.min()) / (im.max() - im.min())) * 255).astype(np.uint8)
                        else:
                            im = (((im-norm[0]) / (norm[1])) * 255).astype(np.uint8)
                        im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
                        return np.ascontiguousarray(im, dtype=np.uint8)

                    reg = torch.atan2(output_batch_[0,0],output_batch_[0,1]).cpu().numpy()

                    cv2.imshow("Vector regression", cv2.flip(apply_colormap(reg), 1))
                    cv2.imshow("Center regression", cv2.flip(apply_colormap(center_heatmap[0,0].cpu().numpy(), norm=(0,1)), 1))

                key = cv2.waitKey(50)
                
                if key == 115: # s
                    if args["camera"] == "allied":
                        image = sample["image"]
                        image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
                        Image.fromarray(image).save(os.path.join(args["data_save_path"], "rgb", f"{save_index:04d}.png"))
                    else:
                        Image.fromarray(sample["image"]).save(os.path.join(args["data_save_path"], "rgb", f"{save_index:04d}.png"))
                        np.save(os.path.join(args["data_save_path"], "depth", f"{save_index:04d}.npy"), sample["depth"])

                    print(f"Saved rgb {'and depth' if args['camera'] != 'allied' else ''}for annotation. {save_index}")
                    save_index += 1
                if key == 32: # space
                    Image.fromarray(cropped_image).save(os.path.join(args["competition_save_path"], f"{competition_index}_input.png"))
                    Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)).save(os.path.join(args["competition_save_path"], f"{competition_index}_marked.png"))
                    
                    if raw_input is not None:
                        torch.save(raw_input, os.path.join(args["competition_save_path"], f"{competition_index}_raw.pth"))

                    with open(os.path.join(args["competition_save_path"], f"{competition_index}_corners.csv"), "w") as f:
                        writer = csv.writer(f)

                        for p in points:
                            writer.writerow(p)

                    with open(os.path.join(args["competition_save_path"], f"{competition_index}_vectors.csv"), "w") as f:
                        writer = csv.writer(f)

                        for begining, end in zip(vectors, points):
                            writer.writerow((*begining, *end))

                    competition_index += 1
                    print("Saved in competition format!")

def main():
    import importlib

    module = importlib.import_module("config.model_args")

    args = module.get_args()
    args['checkpoint_path'] = "/opt/model.pth"
    args['center_checkpoint_path'] = "/opt/center_model.pth"
    args['competition_save_path'] = "/opt/competition_output"
    args['data_save_path'] = "/opt/dataset"

    infer = Inference(args)
    infer.initialize()
    infer.run()

if __name__ == "__main__":
    main()