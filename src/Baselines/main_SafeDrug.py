import os
import sys

# 获取脚本所在目录，正确设置路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _src_dir)
sys.path.insert(0, os.path.dirname(_src_dir))

import dill
import logging
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import Adam

from Baselines.SafeDrug import SafeDrug
from utils.util import multi_label_metric, ddi_rate_score, \
    get_n_params, buildMPNN, create_log_id, logging_config, get_grouped_metrics, get_model_path

import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Training settings
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default="", help="User notes")
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    parser.add_argument('-s', '--single', action='store_true', default=False, help="single visit")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default="log0", help='log dir prefix like "log0"')

    parser.add_argument('--model_name', type=str, default="SafeDrug", help="model name")
    parser.add_argument('--dataset', type=str, default="mimic-iii", help='dataset')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop after this many epochs without improvement')
    parser.add_argument('--cuda', type=int, default=0, help='which cuda')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of node embedding(randomly initialize)')

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--target_ddi', type=float, default=0.15, help='target ddi')
    parser.add_argument('--kp', type=float, default=0.05, help='coefficient of P signal')
    parser.add_argument('--weight_multi', type=float, default=0.05, help='weight of multilabel_margin_loss')
    args = parser.parse_args()
    return args

# evaluate
@torch.no_grad()
def evaluator(model, data_val, voc_size, epoch, ddi_adj_path, rec_results_path=None):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    rec_results = []
    all_pred_prob = []
    ja_visit = [[] for _ in range(5)]
    f1_visit = [[] for _ in range(5)]
    prauc_visit = [[] for _ in range(5)]
    smm_record_visit = [[] for _ in range(5)]

    for patient in tqdm(data_val, ncols=60, total=len(
        data_val), desc=f"Evaluation"):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        visit_weights_patient = []
        if args.test:
            all_diseases, all_procedures, all_medications = [], [], []

        for adm_idx, adm in enumerate(patient):  # every admission
            if args.test:
                all_diseases.append(adm[0])
                all_procedures.append(adm[1])
                all_medications.append(adm[2])
            target_output, _ = model(patient[:adm_idx + 1])  # pre

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)
            visit_weights_patient.append(adm[3])

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            all_pred_prob.append(list(y_pred_tmp))
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)  # 所有patient的y prediction
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        if args.test:
            if len(patient) < 5:
                visit_idx = len(patient) - 1
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

    if args.test and rec_results_path is not None:
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
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)  # if (epoch != 0) | (mode=='test')  else 0.0

    get_grouped_metrics(ja, visit_weights)

    logging.info(
        f'''Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4}, AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, AVG_MED: {med_cnt / visit_cnt:.4}''')
    print(
        f'''[Eval] Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4}, AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, AVG_MED: {med_cnt / visit_cnt:.4}''')

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt


def main(args):
    # set logger
    if args.test:
        args.note = f'test of {args.log_dir_prefix}'
    _log_base = os.path.join(_src_dir, 'log')
    log_directory_path = os.path.join(_log_base, args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log' + str(log_save_id) + '_' + args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # load data
    _data_dir = os.path.join(os.path.dirname(_src_dir), 'data', 'output', args.dataset)
    data_path = os.path.join(_data_dir, 'records_final.pkl')
    voc_path = os.path.join(_data_dir, 'voc_final.pkl')
    ddi_adj_path = os.path.join(_data_dir, 'ddi_A_final.pkl')
    ddi_mask_path = os.path.join(_data_dir, 'ddi_mask_H.pkl')
    molecule_path = os.path.join(_data_dir, 'atc3toSMILES.pkl')

    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]  # 2/3 train
    val_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + val_len]
    data_val = data[split_point + val_len:]
    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    if args.single:
        data_train = [[visit] for patient in data_train for visit in patient]
        data_val = [[visit] for patient in data_val for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]

    # model initialization
    model = SafeDrug(args, voc_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint, average_projection)
    logging.info(model)

    # test-only 模式：只加载已有模型并在 test 集上评估
    if args.test:
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb')))
        model.to(device=device)
        logging.info(f'load model from {model_path}')
        print(f'load model from {model_path}')
        rec_results_path = save_dir + '/' + 'rec_results'
        evaluator(model, data_test, voc_size, 0, ddi_adj_path, rec_results_path)
        return

    # train and validation
    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    logging.info(f'Optimizer: {optimizer}')
    print(f'n_parameters: {get_n_params(model)}')
    print(f'Optimizer: {optimizer}')

    EPOCH = 200
    best_epoch, best_ja = 0, 0
    # 初始化这些变量，防止极端情况下未赋值
    best_prauc, best_ddi_rate, best_avg_med = 0.0, 0.0, 0.0
    best_model_state = deepcopy(model.state_dict())

    for epoch in range(EPOCH):
        epoch += 1
        header = (
            f'\nepoch {epoch} --------------------------model_name={args.model_name}, '
            f'dataset={args.dataset}, lr={args.lr}, target_ddi={args.target_ddi}, kp={args.kp}, '
            f'weight_multi={args.weight_multi}, logger={log_save_id}'
        )
        print(header)
        logging.info(header)
        model.train()
        loss_train, loss_val = 0, 0
        for step, patient in tqdm(enumerate(data_train), ncols=60, desc="Training",
                                  total=len(data_train)):  # every patient
            for idx, adm in enumerate(patient):
                seq_input = patient[:idx + 1]
                result, loss_ddi = model(seq_input)
                loss = loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_train += loss.item() / len(data_train)  # 用于记录每个epoch的总loss

        with torch.no_grad():
            for step, patient in tqdm(enumerate(data_val), ncols=60, desc="Val loss",
                                      total=len(data_val)):  # every patient
                for idx, adm in enumerate(patient):  # every admission
                    seq_input = patient[:idx + 1]  # 前T次数据输入
                    result, loss_ddi = model(seq_input)
                    loss = loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device)
                    loss_val += loss.item() / len(data_val)  # 用于记录每个epoch的总loss

        logging.info(f'loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}')
        print(f'loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}')

        # evaluation on validation set
        ddi_rate, ja, prauc, avg_f1, avg_med = evaluator(
            model, data_val, voc_size, epoch, ddi_adj_path)

        # save best epoch (根据 JA 选择最佳模型)
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_prauc, best_ddi_rate, best_avg_med = ja, prauc, ddi_rate, avg_med
            best_model_state = deepcopy(model.state_dict())
        logging.info('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))
        print('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))

        # early stopping
        if epoch - best_epoch > args.early_stop:  # n个epoch内，验证集性能不上升之后就停
            logging.info(f"Early stopping at epoch {epoch}, no improvement for {args.early_stop} epochs.")
            print(f"Early stopping at epoch {epoch}, no improvement for {args.early_stop} epochs.")
            break

    # save best model
    logging.info('Train finished')
    print('Train finished')
    model_save_path = os.path.join(
        save_dir,
        'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, best_ddi_rate)
    )
    torch.save(best_model_state, open(model_save_path, 'wb'))
    logging.info(f'Best model saved to {model_save_path}')
    print(f'Best model saved to {model_save_path}')

    # ========= 训练后立刻在测试集上评估 =========
    logging.info('Start testing on test set with the best model after training...')
    print('Start testing on test set with the best model after training...')
    model.load_state_dict(best_model_state)
    model.to(device=device)

    prev_test_flag = args.test
    args.test = True  # 让 evaluator 进入 test 模式，保存 rec_results / 直方图等
    rec_results_path = os.path.join(save_dir, 'rec_results')
    evaluator(model, data_test, voc_size, 0, ddi_adj_path, rec_results_path)
    args.test = prev_test_flag


def loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device):
    loss_bce_target = np.zeros((1, voc_size[2]))
    loss_bce_target[:, adm[2]] = 1

    loss_multi_target = np.full((1, voc_size[2]), -1)
    for idx, item in enumerate(adm[2]):
        loss_multi_target[0][idx] = item

    loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
    loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))
    result = F.sigmoid(result).detach().cpu().numpy()[0]
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    y_label = np.where(result == 1)[0]
    current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)

    if current_ddi_rate <= args.target_ddi:
        loss = (1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi
    else:
        beta = max(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
        loss = beta * ((1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi) + (1 - beta) * loss_ddi
    return loss


def plot_hist(all_y_pred, save_path):
    y_pred = [item for sublist in all_y_pred for item in sublist]
    hist, _ = np.histogram(y_pred, bins=10, range=(0, 1))
    logging.info(f'hist: {hist}')
    plt.hist(y_pred)
    plt.savefig(save_path)


def plot_bar(all_bi_emb_hist, save_path):
    logging.info(f'all_bi_emb_hist: {all_bi_emb_hist}')
    x = range(-50, 50, 1)
    plt.bar(x, all_bi_emb_hist)
    plt.savefig(save_path)


if __name__ == '__main__':
    sys.path.append("..")
    torch.manual_seed(1203)
    np.random.seed(2048)
    args = get_args()
    main(args)
