import os
import sys

# 获取脚本所在目录，正确设置路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _src_dir)
sys.path.insert(0, os.path.dirname(_src_dir))

from copy import deepcopy
import dill
import numpy as np
import argparse
from sklearn.metrics import roc_curve
from torch.optim import RMSprop
from tqdm import tqdm
import torch
import time
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Baselines.MICRON import MICRON
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_model_path, get_grouped_metrics

torch.manual_seed(1203)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--note', type=str, default='', help='note')
parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')

parser.add_argument('-s', '--single', action='store_true', default=False, help="single visit")
parser.add_argument('-t', '--test', action='store_true', default=False, help="test mode")
parser.add_argument('-e', '--eval_after_train', action='store_true', default=False, help="evaluate after training")
parser.add_argument('--model_name', type=str, default='MICRON', help="model name")
parser.add_argument('-l', '--log_dir_prefix', type=str, default="log0", help='log dir prefix like "log0", for model test')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='learning rate')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--cuda', type=int, default=3, help='which cuda')

args = parser.parse_args()

# 用于eval函数中判断是否保存测试结果
test_mode = False

def eval(model, data_eval, voc_size, epoch, ddi_adj_path, val=0, threshold1=0.8, threshold2=0.2, rec_results_path=None):
    global test_mode
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    med_cnt, visit_cnt = 0, 0
    label_list, prob_list = [], []

    rec_results = []
    all_pred_prob = []
    ja_visit = [[] for _ in range(5)]
    f1_visit = [[] for _ in range(5)]
    prauc_visit = [[] for _ in range(5)]
    smm_record_visit = [[] for _ in range(5)]

    for step, input in tqdm(enumerate(data_eval), ncols=60, total=len(
        data_eval), desc=f"Evaluation"):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        visit_weights_patient = []
        if test_mode:
            all_diseases, all_procedures, all_medications = [], [], []

        for adm_idx, adm in enumerate(input):
            if test_mode:
                all_diseases.append(adm[0])
                all_procedures.append(adm[1])
                all_medications.append(adm[2])
            if adm_idx == 0:
                representation_base, _, _, _, _ = model(input[:adm_idx+1])

                y_old = np.zeros(voc_size[2])
                y_old[adm[2]] = 1

                y_gt_tmp = np.zeros(voc_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp)
                label_list.append(y_gt_tmp)
                visit_weights_patient.append(adm[3])

                y_pred_tmp = F.sigmoid(representation_base).detach().cpu().numpy()[0]
                all_pred_prob.append(list(y_pred_tmp))
                y_pred_prob.append(y_pred_tmp)
                prob_list.append(y_pred_tmp)

                y_pred_tmp[y_pred_tmp>=0.5] = 1
                y_pred_tmp[y_pred_tmp<0.5] = 0
                y_pred.append(y_pred_tmp)

                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))

                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)

            elif adm_idx >= 1:
                y_gt_tmp = np.zeros(voc_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp)
                label_list.append(y_gt_tmp)

                visit_weights_patient.append(adm[3])

                _, _, residual, _, _ = model(input[:adm_idx+1])
                # prediction prod
                representation_base += residual
                y_pred_tmp = F.sigmoid(representation_base).detach().cpu().numpy()[0]
                all_pred_prob.append(list(y_pred_tmp))
                y_pred_prob.append(y_pred_tmp)
                prob_list.append(y_pred_tmp)

                y_old[y_pred_tmp>=threshold1] = 1
                y_old[y_pred_tmp<threshold2] = 0
                y_pred.append(y_old)

                # prediction label
                y_pred_label_tmp = np.where(y_old == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))
                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        if test_mode:
            if len(input) < 5:
                visit_idx = len(input) - 1
            else:
                visit_idx = 4
            ja_visit[visit_idx].append(adm_ja)
            f1_visit[visit_idx].append(adm_avg_f1)
            prauc_visit[visit_idx].append(adm_prauc)
            smm_record_visit[visit_idx].append(y_pred_label)
            records = [all_diseases, all_procedures, all_medications, y_pred_label, visit_weights_patient, [adm_ja]]
            rec_results.append(records)
                
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        visit_weights.append(np.max(visit_weights_patient))

    if test_mode:
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
        dill.dump(rec_results, open(rec_results_file, 'wb'))
        plot_path = rec_results_path + '/' + 'pred_prob.jpg'
        plot_hist(all_pred_prob, plot_path)
        ja_result_file = rec_results_path + '/' + 'ja_result.pkl'
        dill.dump(ja_visit, open(ja_result_file, 'wb'))
        
        # 输出每个visit分组的详细指标
        print("\n========== 按就诊次数分组的指标 ==========")
        logging.info("\n========== 按就诊次数分组的指标 ==========")
        for i in range(5):
            n_samples = len(ja_visit[i])
            if n_samples > 0:
                ja_mean = np.mean(ja_visit[i])
                f1_mean = np.mean(f1_visit[i])
                prauc_mean = np.mean(prauc_visit[i])
                ddi_v = ddi_rate_score(smm_record_visit[i], path=ddi_adj_path)
            else:
                ja_mean = f1_mean = prauc_mean = ddi_v = 0.0
            msg = f"{i+1}visit (n={n_samples:4d}): JA={ja_mean:.4f}, F1={f1_mean:.4f}, PRAUC={prauc_mean:.4f}, DDI={ddi_v:.4f}"
            print(msg)
            logging.info(msg)
        
    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)

    get_grouped_metrics(ja, visit_weights)

    logging.info('Jaccard: {:.4}, DDI Rate: {:.4}, PRAUC:{:.4}  AVG_F1: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4f}, AVG_MED: {:.4}'.format(
        np.mean(ja), ddi_rate, np.mean(prauc), np.mean(avg_f1), np.mean(avg_p), np.mean(avg_r), med_cnt / visit_cnt
    ))

    if val == 0:
        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt
    else:
        return np.array(label_list), np.array(prob_list)


def run_test(model, data_eval, data_test, voc_size, ddi_adj_path, save_dir, device):
    """运行测试的通用函数"""
    global test_mode
    test_mode = True
    
    model.to(device=device)
    model.eval()
    
    tic = time.time()
    rec_results_path = save_dir + '/' + 'rec_results'
    label_list, prob_list = eval(model, data_eval, voc_size, epoch=0, ddi_adj_path=ddi_adj_path, val=1, rec_results_path=rec_results_path)

    threshold1, threshold2 = [], []
    for i in range(label_list.shape[1]):
        _, _, boundary = roc_curve(label_list[:, i], prob_list[:, i], pos_label=1)
        threshold1.append(min(0.9, max(0.5, boundary[max(0, round(len(boundary) * 0.05) - 1)])))
        threshold2.append(max(0.1, min(0.5, boundary[min(round(len(boundary) * 0.95), len(boundary) - 1)])))
    print(np.mean(threshold1), np.mean(threshold2))
    threshold1 = np.ones(voc_size[2]) * np.mean(threshold1)
    threshold2 = np.ones(voc_size[2]) * np.mean(threshold2)
    eval(model, data_test, voc_size, epoch=0, ddi_adj_path=ddi_adj_path, val=1, threshold1=threshold1, threshold2=threshold2, rec_results_path=rec_results_path)
    logging.info('test time: {}'.format(time.time() - tic))


def main():
    global test_mode
    
    # set logger
    if args.test:
        args.note = f'test of {args.log_dir_prefix}'
        test_mode = True
    _log_base = os.path.join(_src_dir, 'log')
    log_directory_path = os.path.join(_log_base, args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # load data
    _data_dir = os.path.join(os.path.dirname(_src_dir), 'data', 'output', args.dataset)
    data_path = os.path.join(_data_dir, 'records_final.pkl')
    voc_path = os.path.join(_data_dir, 'voc_final.pkl')
    ddi_adj_path = os.path.join(_data_dir, 'ddi_A_final.pkl')
    device = torch.device('cuda:{}'.format(args.cuda))

    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb')) 

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    np.random.seed(1203)
    np.random.shuffle(data)

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point:split_point + eval_len]
    data_test = data[split_point+eval_len:]
    if args.single:
        data_train = [[visit] for patient in data_train for visit in patient]
        data_eval = [[visit] for patient in data_eval for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = MICRON(voc_size, ddi_adj, emb_dim=args.dim, device=device)

    # 仅测试模式
    if args.test:
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb'), map_location=device))
        logging.info("load model from %s", model_path)
        run_test(model, data_eval, data_test, voc_size, ddi_adj_path, save_dir, device)
        return 

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = RMSprop(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # start iterations
    best_epoch, best_ja = 0, 0
    best_ddi_rate = 0

    weight_list = [[0.25, 0.25, 0.25, 0.25]]

    EPOCH = 40
    for epoch in range(EPOCH):
        epoch += 1
        tic = time.time()
        logging.info('epoch {} --------------------------'.format(epoch))
        
        sample_counter = 0
        mean_loss = np.array([0, 0, 0, 0])

        model.train()
        for step, input in tqdm(enumerate(data_train), ncols=60, desc="Training", total=len(data_train)):
            loss = 0
            for adm_idx, adm in enumerate(input):
                seq_input = input[:adm_idx+1]

                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_bce_target_last = np.zeros((1, voc_size[2]))
                loss_bce_target_last[:, input[adm_idx-1][2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                loss_multi_target_last = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(input[adm_idx-1][2]):
                    loss_multi_target_last[0][idx] = item

                result, result_last, _, loss_ddi, loss_rec = model(seq_input)

                loss_bce = 0.75 * F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device)) + \
                    (1 - 0.75) * F.binary_cross_entropy_with_logits(result_last, torch.FloatTensor(loss_bce_target_last).to(device))
                loss_multi = 5e-2 * (0.75 * F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)) + \
                    (1 - 0.75) * F.multilabel_margin_loss(F.sigmoid(result_last), torch.LongTensor(loss_multi_target_last).to(device)))

                y_pred_tmp = F.sigmoid(result).detach().cpu().numpy()[0]
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_label = np.where(y_pred_tmp == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)
                
                if sample_counter == 0:
                    lambda1, lambda2, lambda3, lambda4 = weight_list[-1]
                else:
                    current_loss = np.array([loss_bce.detach().cpu().numpy(), loss_multi.detach().cpu().numpy(), loss_ddi.detach().cpu().numpy(), loss_rec.detach().cpu().numpy()])
                    current_ratio = (current_loss - np.array(mean_loss)) / np.array(mean_loss)
                    instant_weight = np.exp(current_ratio) / sum(np.exp(current_ratio))
                    lambda1, lambda2, lambda3, lambda4 = instant_weight * 0.75 + np.array(weight_list[-1]) * 0.25
                    weight_list.append([lambda1, lambda2, lambda3, lambda4])
                mean_loss = (mean_loss * (sample_counter - 1) + np.array([loss_bce.detach().cpu().numpy(), \
                    loss_multi.detach().cpu().numpy(), loss_ddi.detach().cpu().numpy(), loss_rec.detach().cpu().numpy()])) / sample_counter
                if current_ddi_rate > 0.15:
                    loss += lambda1 * loss_bce + lambda2 * loss_multi + \
                                 lambda3 * loss_ddi +  lambda4 * loss_rec
                else:
                    loss += lambda1 * loss_bce + lambda2 * loss_multi + \
                                lambda4 * loss_rec

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        ddi_rate, ja, prauc, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch, ddi_adj_path)

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            best_ddi_rate = ddi_rate
            best_model_state = deepcopy(model.state_dict()) 

        logging.info(f'best_epoch: {best_epoch}, best_ja: {best_ja:.4f}\n')

    logging.info('Train finished')
    
    # 保存最佳模型
    model_save_path = os.path.join(save_dir, 'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, best_ddi_rate))
    torch.save(best_model_state, open(model_save_path, 'wb'))
    logging.info(f'Model saved to {model_save_path}')
    
    # 训练后直接测试
    if args.eval_after_train:
        logging.info('\n' + '='*50)
        logging.info('Starting evaluation after training...')
        logging.info('='*50)
        
        # 加载最佳模型状态
        model.load_state_dict(best_model_state)
        run_test(model, data_eval, data_test, voc_size, ddi_adj_path, save_dir, device)
    

def plot_hist(all_y_pred, save_path):
    y_pred =  [item for sublist in all_y_pred for item in sublist]
    hist, _ = np.histogram(y_pred, bins = 10, range = (0, 1))
    logging.info(f'hist: {hist}')
    plt.hist(y_pred)
    plt.savefig(save_path)

if __name__ == '__main__':
    main()
