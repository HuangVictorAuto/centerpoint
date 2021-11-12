# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.
# Edited by Huang Yayun to add multi frame dataset function, the waymo_utils.py, init.py, cfgs also need to be updated accordingly.

import os
import pickle
import copy
from typing import Sequence
import numpy as np
from tensorflow._api.v2 import data
import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate

'''
    以datasettemplate为模板，创建各种数据集的dataset，主要涉及到：
    1. 根据train/val/test，提取信息，保存到info.pkl中
    2. 读取info.pkl，get_item，输入到dataloader中
    3. 得到prediction后，转化格式，得到predict_anno
    4. 从info.pkl中得到gt_anno，和predict_anno一起输入到eval中，进行测试
'''

class WaymoDataset_x_sweeps(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sweeps = self.dataset_cfg.NUM_SWEEPS
        self.infos = []
        self.include_waymo_data_multi_sweeps(self.mode)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
    

    def include_waymo_data_multi_sweeps(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('{0}_{1}_sweeps.pkl'.format(sequence_name,self.num_sweeps))

            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

    # 因为waymodataset有两版数据
    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if '_with_camera_labels' not in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file)[:-9] + '_with_camera_labels.tfrecord')
        if '_with_camera_labels' in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))

        return sequence_file


    """
        在对waymo数据进行处理之后，得到了单帧的点云，每一个scene的文件夹，每一个scene对应的pkl文件
        通过这个函数，输出，多帧点云的信息，保存在pkl文件中
        
    """
    # 从raw_data中读取信息，关键是知道waymo raw data的构成，输出我们期望的标准的data和info
    # tf-record,合计1000/150，每个tf-record 20s,10Hz,合计230k个frame,每个record，分解出~200frame
    # 每个frame中包含很多信息:传感器标定信息、图像信息、点云信息、2d box、3d box，这里主要存储了点，3d box的信息
    def get_infos_multi_sweeps(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1,num_sweeps=1):

        import concurrent.futures as futures
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))
       
        process_single_sequence = partial(
            waymo_utils.process_single_sequence_multi_sweeps,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label,num_sweeps=num_sweeps
        )

        # process_single_sequence(sample_sequence_file_list[0])
        with futures.ThreadPoolExecutor(num_workers) as executor:
            sequence_infos = list(tqdm(executor.map(process_single_sequence, self.sample_sequence_list),
                                       total=len(self.sample_sequence_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos
    

    # 根据sweep的info信息，返回处理好的点和时间
    def read_single_waymo_sweep(self,sweep):
        sequence_name = sweep['point_cloud']['lidar_sequence']
        sample_idx = sweep['point_cloud']['sample_idx']
        
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)

        points_all, NLZ_flag = point_features[:,0:4],point_features[:,5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:,3] = np.tanh(points_all[:,3])

        points_sweep = points_all.T # 5 x N

        nbr_points = points_sweep.shape[1]

        if sweep["transform_matrix"] is not None:
            points_sweep[:3, :] = sweep["transform_matrix"].dot( 
                np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
            )[:3, :]
            
        times_sweep = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

        return points_sweep.T,times_sweep.T
    


    # 从npy文件中，读取点云信息，去除掉NLZ的点，返回[x,y,z,intensity,time]
    def get_lidar_multi_sweeps(self, sequence_name,sample_idx,sweeps):
        
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)

        points_all, NLZ_flag = point_features[:,0:4],point_features[:,5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:,3] = np.tanh(points_all[:,3])

        sweep_points_list = [points_all]
        
        sweep_times_list = [np.zeros((points_all.shape[0],1))]

        for i in range(self.num_sweeps-1):
            sweep = sweeps[i]
            points_sweep,times_sweep = self.read_single_waymo_sweep(sweep)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)
        
        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
    
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)
    # 输入到网络的信息
    # input_dict:{'points','frame_id','gt_names','gt_boxes','num_points_in_gt'}
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        sweeps = info['sweeps']
        points = self.get_lidar_multi_sweeps(sequence_name, sample_idx,sweeps)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }
        # 去除unknown标签的数据
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })
        # 输入信息到prepare_data中，进行数据增强、点云特征编码、前处理
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    # 从网络输出的结果中，转换格式，生成标准的prediction  
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        # prediction模板，{'name','score','boxes_lidar'}
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict
        # gpu to cpu，输出pred_dict
        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict
        # annos {'name','score','boxes_lidar','frame_id','metadata'}
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos
    
    '''
        evaluation,可以选用kitti的eval matrix，也可以选用waymo的matrix
        det_annos {'name','score','boxes_lidar','frame_id','metadata'}
        gt_annos {'name','difficulty','dimensions','location','heading_angles','object_id','tracking_difficulty','num_points_in_gt','gt_boxes_lidar'}
        调用waymo官方的evaluation的script
    '''
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict
        # 输入dt_annos,gt_annos，进行evaulate
        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
    # 从training数据中，提取object对应的点云，用于后面的数据增强
    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None, num_sweeps = 3):
        database_save_path = save_path / ('pcdet_gt_database_%s_sampled_%d' % (split, sampled_interval))
        db_info_save_path = save_path / ('pcdet_waymo_dbinfos_%s_sampled_%d.pkl' % (split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            sweeps = info['sweeps']
            points = self.get_lidar_multi_sweeps(sequence_name, sample_idx,sweeps)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


# 初始化waymo数据类,读取id文件，分别存储train info,val info和ground truth data和info
def create_waymo_infos_multi_sweeps(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=multiprocessing.cpu_count(),num_sweeps = 1):
    dataset = WaymoDataset_x_sweeps(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('waymo_infos_{0}_{1}sweeps.pkl'.format(train_split,num_sweeps))
    val_filename = save_path / ('waymo_infos_{0}_{1}sweeps.pkl'.format(val_split,num_sweeps))

    # print('---------------Start to generate data infos---------------')

    # dataset.set_split(train_split)
    # waymo_infos_train = dataset.get_infos_multi_sweeps(
    #     raw_data_path=data_path / raw_data_tag,
    #     save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
    #     sampled_interval=dataset_cfg.SAMPLED_INTERVAL['train'],
    #     num_sweeps = num_sweeps
    # )
    # with open(train_filename, 'wb') as f:
    #     pickle.dump(waymo_infos_train, f)
    # print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    # dataset.set_split(val_split)
    # waymo_infos_val = dataset.get_infos_multi_sweeps(
    #     raw_data_path=data_path / raw_data_tag,
    #     save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
    #     sampled_interval=dataset_cfg.SAMPLED_INTERVAL['test'],
    #     num_sweeps = num_sweeps
    # )
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(waymo_infos_val, f)
    # print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=10,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'],num_sweeps = num_sweeps
    )

    print('---------------Data preparation Done---------------')


# 单独运行，读取tf-record文件中的raw data，存储processed data和info文件
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    args = parser.parse_args()

    if args.func == 'create_waymo_infos_multi_sweeps':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_waymo_infos_multi_sweeps(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG,
            num_sweeps = dataset_cfg.NUM_SWEEPS
        )