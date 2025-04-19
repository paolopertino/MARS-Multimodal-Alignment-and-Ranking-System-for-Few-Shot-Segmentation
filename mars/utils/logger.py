r""" Logging during training/testing. Adapted from Matcher"""
import datetime
import logging
import os
import subprocess as sp

from threading import Timer

from comet_ml import Experiment
import torch
from tensorboardX import SummaryWriter


class AverageMeter:
    r""" Stores loss, evaluation results """

    def __init__(self, dataset, device='cpu'):
        self.device = device
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids

        if self.benchmark == 'pascal5i':
            self.class_ids_interest = [
                i-1 for i in self.class_ids_interest]  # 1-index to 0-index

        self.class_ids_interest = torch.tensor(
            self.class_ids_interest).to(self.device)

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'pascal5i':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000
        elif self.benchmark == 'paco_part':
            self.nclass = 448
        elif self.benchmark == 'pascal_part':
            self.nclass = 100
        elif self.benchmark == 'lvis':
            self.nclass = 1203

        self.intersection_buf = torch.zeros(
            [2, self.nclass]).float().to(self.device)
        self.union_buf = torch.zeros([2, self.nclass]).float().to(self.device)

        # Adding support for known bad predictions
        # If no bad predictions are provided, these vectors
        # won't be used.
        self.class_ids_known_bad = []
        self.intersection_buf_known_bad = torch.zeros(
            [2, self.nclass]).float().to(self.device)
        self.union_buf_known_bad = torch.zeros(
            [2, self.nclass]).float().to(self.device)
        self.loss_buf_known_bad = []

        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
            torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        # In order not to have a too long list of categories in the log message
        return miou, fb_iou, iou[1][:min(len(iou[1]), 20)]

    def update_bad_preds(self, inter_b, union_b, class_id, loss):
        if class_id not in self.class_ids_known_bad:
            self.class_ids_known_bad.append(class_id)

        self.intersection_buf_known_bad.index_add_(
            1, class_id, inter_b.float())
        self.union_buf_known_bad.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf_known_bad.append(loss)

    def compute_iou_bad_preds(self):
        iou = self.intersection_buf_known_bad.float() / \
            torch.max(torch.stack(
                [self.union_buf_known_bad, self.ones]), dim=0)[0]
        iou = iou.index_select(1, torch.tensor(
            self.class_ids_known_bad).to(self.device))
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf_known_bad.index_select(1, torch.tensor(self.class_ids_known_bad).to(self.device)).sum(dim=1) /
                  self.union_buf_known_bad.index_select(1, torch.tensor(self.class_ids_known_bad).to(self.device)).sum(dim=1)).mean() * 100

        # In order not to have a too long list of categories in the log message
        return miou, fb_iou, iou[1][:min(len(iou[1]), 20)]

    def write_result(self, split, epoch):
        iou, fb_iou, cats_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou
        for cat, cat_iou in enumerate(cats_iou):
            cat_iou = cat_iou * 100
            msg += f' |  {cat}:'+' %5.2f   ' % cat_iou

        msg += '***\n'
        Logger.info(msg)

    def write_result_bad_preds(self, split, epoch):
        iou, fb_iou, cats_iou = self.compute_iou_bad_preds()

        loss_buf = torch.stack(self.loss_buf_known_bad)
        msg = '\n*** %s - Bad Preds' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou
        for cat, cat_iou in enumerate(cats_iou):
            cat_iou = cat_iou * 100
            msg += f' |  {cat}:'+' %5.2f   ' % cat_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou, cats_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            for cat, cat_iou in enumerate(cats_iou):
                cat_iou = cat_iou * 100
                msg += f' |  {cat}:' + ' %5.2f   ' % cat_iou
            print(msg)
            # Logger.info(msg)

    def write_process_bad_preds(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d - Bad Pred] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou, cats_iou = self.compute_iou_bad_preds()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf_known_bad)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            for cat, cat_iou in enumerate(cats_iou):
                cat_iou = cat_iou * 100
                msg += f' |  {cat}:' + ' %5.2f   ' % cat_iou

            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, root='logs'):
        logtime = datetime.datetime.now().__format__('%m%d_%H%M%S')
        logpath = '_TEST_' + logtime

        cls.logpath = os.path.join(root, logpath + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with MARS ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' %
                         (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(
            cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))


class CometLogger():
    """ CometML Logger """

    def __init__(self,
                 comet_api_key: str,
                 project_name: str,
                 workspace: str,
                 average_meter: AverageMeter,
                 ):
        self.key = comet_api_key
        self.project = project_name
        self.workspace = workspace
        self.average_meter = average_meter

        self.exp = None
        self.fold_num = None
        self.current_idx = 0  # Current batch index

    def initialize_experiment(self, args):
        self.fold_num = args.fold
        self.exp = Experiment(
            api_key=self.key, project_name=self.project, workspace=self.workspace)
        self.exp.log_parameters(vars(args))
        self.exp.set_name(args.exp_name)

    def log_metrics(self, batch_idx, split):
        iou, fb_iou, cats_iou = self.average_meter.compute_iou()
        self.exp.log_metric(f'{split}_mIoU', iou, step=batch_idx)
        self.exp.log_metric(f'{split}_FB-IoU', fb_iou, step=batch_idx)
        for cat, cat_iou in enumerate(cats_iou):
            self.exp.log_metric(f'{split}_cat_{cat}_IoU',
                                cat_iou, step=batch_idx)

    def log_metrics_bad_preds(self, bad_preds_results, miou_per_class, miou_overall):
        for result in bad_preds_results:
            self.exp.log_metric(
                f"sample{result['idx']}_class{result['class_id']}_IoU", result['iou'])

        for class_idx, mean_iou in miou_per_class.items():
            self.exp.log_metric(f'class{class_idx}_mIoU', mean_iou)

        self.exp.log_metric('bad_preds_mIoU', miou_overall)

    def log_time_batch(self, time_elapsed_batch, idx):
        self.exp.log_metric(f'time_elapsed_batch_{idx}', time_elapsed_batch)

    def log_avg_time_elapsed(self, time_elapsed_per_batch):
        avg_time_elapsed_per_batch = sum(
            time_elapsed_per_batch) / len(time_elapsed_per_batch)
        self.exp.log_metric('avg_time_elapsed_per_batch',
                            avg_time_elapsed_per_batch)

    def log_image(self, image_data, name, metadata: dict = None):
        self.exp.log_image(image_data, name=name, metadata=metadata)

    def log_figure(self, figure=None, name=None):
        self.exp.log_figure(figure=figure, figure_name=name)

    def end_experiment(self, total_time_elapsed):
        self.exp.log_metric('total_time_elapsed', total_time_elapsed)
        self.exp.end()
