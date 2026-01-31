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
import random
import time  # 新增：用于记录总运行时间

import torch
import torch.nn.functional as F
from torch.optim import AdamW as Optimizer

from Baselines.RAREMed import RAREMed
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_grouped_metrics, \
    get_model_path, get_pretrained_model_path

# Training settings
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='RAREMed', help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')
    parser.add_argument('--early_stop', type=int, default=30, help='early stop after this many epochs without improvement')
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    # 新增：训练结束后立刻在测试集上评估
    parser.add_argument('--train_and_test', action='store_true',
                        help='train the model and then immediately test on the test set in one run')
    # pretrain的模型参数也是利用log_dir_prefix来确定是哪个log里的模型
    parser.add_argument('-l', '--log_dir_prefix', type=str, default=None, help='log dir prefix like "log0", for model test')
    parser.add_argument('-p', '--pretrain_prefix', type=str, default=None, help='log dir prefix like "log0", for finetune')
    parser.add_argument('--cuda', type=int, default=6, help='which cuda')
    # pretrain
    parser.add_argument('-nsp', '--pretrain_nsp', action='store_true', help='whether to use nsp pretrain')
    parser.add_argument('-mask', '--pretrain_mask', action='store_true', help='whether to use mask prediction pretrain')
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='number of pretrain epochs')
    parser.add_argument('--mask_prob', type=float, default=0, help='mask probability')

    parser.add_argument('--embed_dim', type=int, default=512, help='dimension of node embedding')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--nhead', type=int, default=4, help='number of encoder head')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size during training')
    parser.add_argument('--adapter_dim', type=int, default=128, help='dimension of adapter layer')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout probability of transformer encoder')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--weight_multi', type=float, default=0.005, help='weight of multilabel_margin_loss')
    parser.add_argument('--weight_ddi', type=float, default=0.1, help='weight of ddi loss')

    # parameters for ablation study
    parser.add_argument('-s', '--patient_seperate', action='store_true', help='whether to combine diseases and procedures')
    parser.add_argument('-e', '--seg_rel_emb', action='store_false', default=True, help='whether to use segment and relevance embedding layer')
    parser.add_argument('--RAREMed_epoch', type=int, default=50, help='RAREMed epoch')

    args = parser.parse_args()
    return args


def single_visit_metric(y_gt, y_pred, y_pred_prob):
    """计算单个 visit 的指标"""
    from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, recall_score, f1_score
    
    # Jaccard - 使用集合交并比
    gt_set = set(np.where(y_gt == 1)[0])
    pred_set = set(np.where(y_pred == 1)[0])
    
    if len(gt_set) == 0 and len(pred_set) == 0:
        ja = 1.0
    elif len(gt_set) == 0 or len(pred_set) == 0:
        ja = 0.0
    else:
        intersection = len(gt_set & pred_set)
        union = len(gt_set | pred_set)
        ja = intersection / union if union > 0 else 0.0
    
    # PRAUC
    try:
        if np.sum(y_gt) == 0:
            prauc = 0.0
        else:
            prauc = roc_auc_score(y_gt, y_pred_prob)
    except:
        prauc = 0.0
    
    # Precision, Recall, F1
    if np.sum(y_pred) == 0:
        precision = 0.0
    else:
        precision = precision_score(y_gt, y_pred, zero_division=0)
    
    if np.sum(y_gt) == 0:
        recall = 0.0
    else:
        recall = recall_score(y_gt, y_pred, zero_division=0)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return ja, prauc, precision, recall, f1


# evaluate
@torch.no_grad()
def evaluator(args, model, data_val, voc_size, epoch,
              ddi_adj_path, device, rec_results_path=''):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    smm_record = []                 # 统计所有推荐结果，最后一起算一个DDI rate
    med_cnt, visit_cnt = 0, 0       # 统计平均药物数量
    recommended_drugs = set()
    loss_val_bce = loss_val_milti = loss_val_ddi = loss_val_sum = 0
    len_val = len(data_val)

    rec_results = []
    all_pred_prob = []
    ja_visit = [[] for _ in range(5)]
    f1_visit = [[] for _ in range(5)]
    prauc_visit = [[] for _ in range(5)]
    smm_record_visit = [[] for _ in range(5)]

    for batch in tqdm(data_val, ncols=60, total=len(data_val), desc="Evaluation"):
        """
            y_gt:         batch_size * drug_num, binary
            y_pred_prob:  batch_size * drug_num, predict probability, float(0-1)
            y_pred:       batch_size * drug_num, predict label, binary
            y_pred_label: batch_size * pred_num, predicted drug index, int
        """
        batch_size = len(batch)
        y_gt_list, y_pred_list, y_pred_prob_list, y_pred_label_list = [], [], [], []
        batch_weights = []

        all_diseases, all_procedures, all_medications = [], [], []   # if test

        for adm in batch:  # every admission in batch
            if args.test:
                all_diseases.append(adm[0])
                all_procedures.append(adm[1])
                all_medications.append(adm[2])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt_list.append(y_gt_tmp)  # correct medicine
            visit_weights.append(adm[3])
            batch_weights.append(adm[3])

        results, loss_ddi = model(batch)
        loss_bce, loss_multi = loss_func(voc_size, batch, results, device)
        loss_val_bce += loss_bce.item() / len_val
        loss_val_milti += loss_multi.item() / len_val
        loss_val_ddi += loss_ddi.item() / len_val
        y_pred_prob_batch = F.sigmoid(results).detach().cpu().numpy()

        batch_y_pred_label = []  # 用于 smm_record
        for i, target_output in enumerate(y_pred_prob_batch):       # each visit in batch
            # prediction med set
            y_pred_tmp = target_output.copy()
            all_pred_prob.append(list(y_pred_tmp))
            y_pred_prob_list.append(target_output)
            
            y_pred_binary = target_output.copy()
            y_pred_binary[y_pred_binary >= 0.5] = 1
            y_pred_binary[y_pred_binary < 0.5] = 0
            y_pred_list.append(y_pred_binary)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_binary == 1)[0]
            recommended_drugs = set(y_pred_label_tmp) | recommended_drugs
            y_pred_label_list.append(sorted(y_pred_label_tmp))
            batch_y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

            # 修复：为每个 visit 单独计算指标
            visit_ja, visit_prauc, visit_p, visit_r, visit_f1 = single_visit_metric(
                y_gt_list[i], y_pred_binary, target_output)
            
            ja.append(visit_ja)
            prauc.append(visit_prauc)
            avg_p.append(visit_p)
            avg_r.append(visit_r)
            avg_f1.append(visit_f1)

        smm_record.append(batch_y_pred_label)  # 所有batch的y prediction
        
        if args.test:
            # 为每个visit单独统计到对应的visit组
            for i, adm in enumerate(batch):
                # 获取该visit的visit_idx (患者的第几次就诊, 从adm[4]获取)
                patient_visit_idx = adm[4] if len(adm) > 4 else 0
                if patient_visit_idx >= 5:
                    group_idx = 4  # 5次及以上归为第5组
                else:
                    group_idx = patient_visit_idx
                
                ja_visit[group_idx].append(ja[-batch_size + i])
                f1_visit[group_idx].append(avg_f1[-batch_size + i])
                prauc_visit[group_idx].append(prauc[-batch_size + i])
                smm_record_visit[group_idx].append([batch_y_pred_label[i]])
            
            records = [all_diseases, all_procedures, all_medications, y_pred_label_list, batch_weights, ja[-batch_size:]]
            rec_results.append(records)

    if args.test:
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = os.path.join(rec_results_path, 'rec_results.pkl')
        dill.dump(rec_results, open(rec_results_file, 'wb'))
        plot_path = os.path.join(rec_results_path, 'pred_prob.jpg')
        # plot_hist(all_pred_prob, plot_path)
        ja_result_file = os.path.join(rec_results_path, 'ja_result.pkl')
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

    # 现在 ja 和 visit_weights 长度一致了
    get_grouped_metrics(ja, visit_weights)
    logging.info(
        f"Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, "
        f"PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4}, "
        f"AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, "
        f"AVG_MED: {med_cnt / visit_cnt:.4}"
    )

    loss_val_sum = (1 - args.weight_multi) * loss_val_bce + args.weight_multi * loss_val_milti + args.weight_ddi * loss_val_ddi

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
        loss_val_bce,
        loss_val_milti,
        loss_val_ddi,
        loss_val_sum,
    )

@torch.no_grad()
def evaluator_mask(model, data_val, voc_size, epoch, device, mode='pretrain'):
    model.eval()
    loss_val = 0
    dis_ja_list, dis_prauc_list, dis_p_list, dis_r_list, dis_f1_list = [[] for _ in range(5)]
    pro_ja_list, pro_prauc_list, pro_p_list, pro_r_list, pro_f1_list = [[] for _ in range(5)]
    dis_cnt, pro_cnt, visit_cnt = 0, 0, 0
    recommended_drugs = set()
    len_val = len(data_val)
    for batch in tqdm(data_val, ncols=60, desc=mode, total=len_val):
        batch_size = len(batch)
        dis_pred, dis_pred_label = [[] for _ in range(2)]
        pro_pred, pro_pred_label = [[] for _ in range(2)]

        result = model(batch, mode)
        dis_gt = np.zeros((batch_size, voc_size[0]))
        pro_gt = np.zeros((batch_size, voc_size[1]))
        for i in range(batch_size):
            dis_gt[i, batch[i][0]] = 1
            pro_gt[i, batch[i][1]] = 1
        target = np.concatenate((dis_gt, pro_gt), axis=1)
        loss = F.binary_cross_entropy_with_logits(result, torch.tensor(target, device=device))
        loss_val += loss.item()

        dis_logit = result[:, :voc_size[0]]
        pro_logit = result[:, voc_size[0]:]
        dis_pred_prob = F.sigmoid(dis_logit).cpu().numpy()
        pro_pred_prob = F.sigmoid(pro_logit).cpu().numpy()

        visit_cnt += batch_size
        for i in range(batch_size):
            dis_pred_temp = dis_pred_prob[i].copy()
            dis_pred_temp[dis_pred_temp >= 0.5] = 1
            dis_pred_temp[dis_pred_temp < 0.5] = 0
            dis_pred.append(dis_pred_temp)

            dis_pred_label_temp = np.where(dis_pred_temp == 1)[0]
            dis_pred_label.append(sorted(dis_pred_label_temp))
            dis_cnt += len(dis_pred_label_temp)

            pro_pred_temp = pro_pred_prob[i].copy()
            pro_pred_temp[pro_pred_temp >= 0.5] = 1
            pro_pred_temp[pro_pred_temp < 0.5] = 0
            pro_pred.append(pro_pred_temp)

            pro_pred_label_temp = np.where(pro_pred_temp == 1)[0]
            pro_pred_label.append(sorted(pro_pred_label_temp))
            pro_cnt += len(pro_pred_label_temp)

        dis_ja, dis_prauc, dis_avg_p, dis_avg_r, dis_avg_f1 = multi_label_metric(
            np.array(dis_gt), np.array(dis_pred), np.array(dis_pred_prob))
        pro_ja, pro_prauc, pro_avg_p, pro_avg_r, pro_avg_f1 = multi_label_metric(
            np.array(pro_gt), np.array(pro_pred), np.array(pro_pred_prob))

        dis_ja_list.append(dis_ja)
        dis_prauc_list.append(dis_prauc)
        dis_p_list.append(dis_avg_p)
        dis_r_list.append(dis_avg_r)
        dis_f1_list.append(dis_avg_f1)

        pro_ja_list.append(pro_ja)
        pro_prauc_list.append(pro_prauc)
        pro_p_list.append(pro_avg_p)
        pro_r_list.append(pro_avg_r)
        pro_f1_list.append(pro_avg_f1)

        avg_dis_ja = np.mean(dis_ja_list)
        avg_dis_prauc = np.mean(dis_prauc_list)
        avg_dis_p = np.mean(dis_p_list)
        avg_dis_r = np.mean(dis_r_list)
        avg_dis_f1 = np.mean(dis_f1_list)

        avg_pro_ja = np.mean(pro_ja_list)
        avg_pro_prauc = np.mean(pro_prauc_list)
        avg_pro_p = np.mean(pro_p_list)
        avg_pro_r = np.mean(pro_r_list)
        avg_pro_f1 = np.mean(pro_f1_list)

        avg_ja = (avg_dis_ja + avg_pro_ja) / 2
        avg_prauc = (avg_dis_prauc + avg_pro_prauc) / 2
        avg_p = (avg_dis_p + avg_pro_p) / 2
        avg_r = (avg_dis_r + avg_pro_r) / 2
        avg_f1 = (avg_dis_f1 + avg_pro_f1) / 2
        avg_dis_cnt = dis_cnt / visit_cnt
        avg_pro_cnt = pro_cnt / visit_cnt
        avg_cnt = (avg_dis_cnt + avg_pro_cnt) / 2

    logging.info(
        'Epoch {:03d}   Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_CNT: {:.4}'.format(
            epoch, avg_ja, avg_prauc, avg_p, avg_r, avg_f1, avg_cnt))
    logging.info(
        'Epoch {:03d}   DISEASE Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_DIS_CNT: {:.4}'.format(
            epoch, avg_dis_ja, avg_dis_prauc, avg_dis_p, avg_dis_r, avg_dis_f1, avg_dis_cnt))
    logging.info(
        'Epoch {:03d}   PROCEDURE Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_PRO_CNT: {:.4}'.format(
            epoch, avg_pro_ja, avg_pro_prauc, avg_pro_p, avg_pro_r, avg_pro_f1, avg_pro_cnt))
    return loss_val / len_val, avg_ja, avg_prauc, avg_p, avg_r, avg_f1, avg_cnt

@torch.no_grad()
def evaluator_nsp(model, data_val, data, epoch, device, mode='pretrain_nsp'):
    model.eval()
    loss_val = 0
    prc = []
    len_val = len(data_val)
    for batch_data in tqdm(data_val, ncols=60, desc=mode, total=len_val):
        nsp_batch, nsp_target = nsp_batch_data(batch_data, data, neg_sample_rate=1)
        result = model(nsp_batch, mode='pretrain_nsp')  # 形状应为 [N] 或 [N,1]
        target_tensor = torch.tensor(nsp_target, device=device, dtype=torch.float32)

        # 二分类：使用 BCEWithLogitsLoss
        if result.dim() == 2 and result.size(1) == 1:
            result_for_loss = result.view(-1)
        else:
            result_for_loss = result

        loss = F.binary_cross_entropy_with_logits(result_for_loss, target_tensor)
        loss_val += loss.item()

        # 计算准确率
        result_np = result_for_loss.detach().cpu().numpy()
        preds = (result_np > 0.5).astype(np.int32)
        prc.append(np.mean(preds == np.array(nsp_target)))

    return np.mean(prc), loss_val

def random_mask_word(seq, vocab, mask_prob=0.15):
    mask_idx = vocab.word2idx['[MASK]']
    for i, _ in enumerate(seq):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob
            # 80% randomly change token to mask token
            if prob < 0.8:
                seq[i] = mask_idx
            # 10% randomly change token to random token
            elif prob < 0.9:
                seq[i] = random.choice(list(vocab.word2idx.items()))[1]
            else:
                pass
        else:
            pass
    return seq

# mask batch data
def mask_batch_data(batch_data, diag_voc, pro_voc, mask_prob):
    masked_data = []
    for visit in batch_data:
        diag = random_mask_word(visit[0], diag_voc, mask_prob)
        pro = random_mask_word(visit[1], pro_voc, mask_prob)
        masked_data.append([diag, pro])
    return masked_data

def nsp_batch_data(batch_data, data, neg_sample_rate=1):
    nsp_batch = []
    nsp_target = []
    for visit in batch_data:
        nsp_batch.append(visit)
        nsp_target.append(1)
        for _ in range(neg_sample_rate):
            neg_visit = random.choice(data)
            while neg_visit[1] == visit[1]:
                neg_visit = random.choice(data)
            if random.random() < 0.5:
                nsp_batch.append([visit[0], neg_visit[1]])
                nsp_target.append(0)
            else:
                nsp_batch.append([neg_visit[0], visit[1]])
                nsp_target.append(0)
    return nsp_batch, nsp_target

def main(args):
    # set logger
    if args.test:
        args.note = 'test of ' + args.log_dir_prefix
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

    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    def add_word(word, voc_single):
        voc_single.word2idx[word] = len(voc_single.word2idx)
        voc_single.idx2word[len(voc_single.idx2word)] = word
        return voc_single

    add_word('[MASK]', diag_voc)
    add_word('[MASK]', pro_voc)

    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    val_len = int(len(data[split_point:]) / 2)
    data_val = data[split_point:split_point + val_len]
    data_pretrain = data[:split_point + val_len]
    data_test = data[split_point + val_len:]

    # convert data into single visit format
    # 为每个visit添加visit_idx (该患者的第几次就诊)
    def flatten_with_visit_idx(data):
        result = []
        for patient in data:
            for visit_idx, visit in enumerate(patient):
                # 将visit_idx添加到visit数据中 (作为第5个元素)
                if len(visit) == 4:
                    result.append(list(visit) + [visit_idx])
                else:
                    result.append(list(visit[:4]) + [visit_idx])
        return result
    
    data_pretrain = [visit for patient in data_pretrain for visit in patient]
    data_train = [visit for patient in data_train for visit in patient]
    data_val = [visit for patient in data_val for visit in patient]
    data_test = flatten_with_visit_idx(data_test)  # 只有test需要visit分组

    # batchify data
    def batchify(data_list, batch_size):
        return [data_list[i:min(i + batch_size, len(data_list))] for i in range(0, len(data_list), batch_size)]

    data_train = batchify(data_train, args.batch_size)
    data_val = batchify(data_val, args.batch_size)
    data_test = batchify(data_test, args.batch_size)

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    # model initialization
    model = RAREMed(args, voc_size, ddi_adj)
    logging.info(model)

    # test
    if args.test:
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb'), map_location=device))
        model.to(device=device)
        logging.info("load model from %s", model_path)
        rec_results_path = os.path.join(save_dir, 'rec_results')

        evaluator(args, model, data_test, voc_size, 0, ddi_adj_path, device, rec_results_path)
        return
    else:
        # 不使用TensorBoard，writer仅作为占位符
        writer = None

    # train and validation
    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logging.info(f'Optimizer: {optimizer}')

    if args.dataset == 'mimic-iii':
        if args.pretrain_nsp:
            main_nsp(args, model, optimizer, writer, data_pretrain, data_train, data_val, device, save_dir, log_save_id)
        if args.pretrain_mask:
            main_mask(args, model, optimizer, writer, diag_voc, pro_voc, data_train, data_val,
                      voc_size, device, save_dir, log_save_id)
    else:
        if args.pretrain_nsp:
            main_nsp(args, model, optimizer, writer, data_pretrain, data_train, data_val, device, save_dir, log_save_id)
        if args.pretrain_mask:
            main_mask(args, model, optimizer, writer, diag_voc, pro_voc, data_train, data_val,
                      voc_size, device, save_dir, log_save_id)

    if not (args.pretrain_mask or args.pretrain_nsp) and args.pretrain_prefix is not None:
        # if not pretrain, load pretrained model; else, train from scratch
        pretrained_model_path = get_pretrained_model_path(log_directory_path, args.pretrain_prefix)
        load_pretrained_model(model, pretrained_model_path)

    EPOCH = args.RAREMed_epoch
    best_epoch, best_ja = 0, 0
    best_ddi_rate = 0.0
    best_model_state = None

    # 使用 tqdm 包裹 epoch 循环，但不改变原有逻辑
    for epoch in tqdm(range(EPOCH), ncols=60, desc="RAREMed_train", total=EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} --------------------------model_name={args.model_name}, logger={log_save_id}')

        # finetune
        model.train()
        loss_train_bce, loss_train_multi, loss_train_ddi, loss_train_all = 0, 0, 0, 0

        for step, patient in tqdm(enumerate(data_train), ncols=60, desc="finetune", total=len(data_train)):
            result, loss_ddi = model(patient)
            loss_bce, loss_multi = loss_func(voc_size, patient, result, device)
            loss_all = (1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi + args.weight_ddi * loss_ddi
            loss_final = loss_all / args.batch_size
            loss_final.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_train_bce += loss_bce.item() / len(data_train)
            loss_train_multi += loss_multi.item() / len(data_train)
            loss_train_ddi += loss_ddi.item() / len(data_train)
            loss_train_all += loss_all.item() / len(data_train)

        # evaluation
        (
            ddi_rate,
            ja,
            prauc,
            avg_f1,
            avg_med,
            loss_val_bce,
            loss_val_multi,
            loss_val_ddi,
            loss_val_all,
        ) = evaluator(args, model, data_val, voc_size, epoch, ddi_adj_path, device)

        logging.info(
            f"loss_train_all:{loss_train_all:.4f}, loss_train_bce: {loss_train_bce:.4f}, "
            f"loss_train_multi:{loss_train_multi:.4f}, loss_train_ddi:{loss_train_ddi:.4f}\n"
            f"loss_val_all:  {loss_val_all:.4f}, loss_val_bce:  {loss_val_bce:.4f}, "
            f"loss_val_multi:  {loss_val_multi:.4f}, loss_val_ddi:  {loss_val_ddi:.4f}"
        )
        tensorboard_write(
            writer, ja, prauc, ddi_rate, avg_med, epoch,
            loss_train_bce, loss_train_multi, loss_train_ddi, loss_train_all,
            loss_val_bce, loss_val_multi, loss_val_ddi, loss_val_all,
        )

        # save best epoch
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_ddi_rate = ja, ddi_rate
            best_model_state = deepcopy(model.state_dict())
        logging.info(f'best_epoch: {best_epoch}, best_ja: {best_ja:.4f}\n')

        if epoch - best_epoch > args.early_stop:
            break

    # save the best model
    logging.info('Train finished')
    if best_model_state is not None:
        best_model_file = os.path.join(
            save_dir,
            'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, best_ddi_rate)
        )
        torch.save(best_model_state, open(best_model_file, 'wb'))
        logging.info('Best model saved to %s', best_model_file)
    else:
        logging.info('No best model state saved (training may have ended before any improvement).')

    # 如果指定了 --train_and_test，则训练完成后立刻在测试集上评估 best model
    if args.train_and_test and best_model_state is not None:
        logging.info('Start testing with the best model on the test set...')
        model.load_state_dict(best_model_state)
        model.to(device=device)
        args.test = True
        rec_results_path = os.path.join(save_dir, 'rec_results')
        evaluator(args, model, data_test, voc_size, 0, ddi_adj_path, device, rec_results_path)

def main_mask(args, model, optimizer, writer, diag_voc, pro_voc, data_train, data_val, voc_size, device, save_dir, log_save_id):
    epoch_mask = 0
    best_epoch_mask, best_ja_mask = 0, 0
    EPOCH = args.pretrain_epochs
    # 用 tqdm 包裹 mask 预训练 epoch 循环
    for epoch in tqdm(range(EPOCH), ncols=60, desc="pretrain_mask_epoch", total=EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} --------------------------model_name={args.model_name}, logger={log_save_id}, mode=pretrain_mask')

        # mask pretrain
        model.train()
        epoch_mask += 1
        loss_train = 0
        for batch in tqdm(data_train, ncols=60, desc="pretrain_mask", total=len(data_train)):
            batch_size = len(batch)
            if args.mask_prob > 0:
                masked_batch = mask_batch_data(batch, diag_voc, pro_voc, args.mask_prob)
            else:
                masked_batch = batch
            result = model(masked_batch, mode='pretrain_mask')
            bce_target_dis = np.zeros((batch_size, voc_size[0]))
            bce_target_pro = np.zeros((batch_size, voc_size[1]))

            for i in range(batch_size):
                bce_target_dis[i, batch[i][0]] = 1
                bce_target_pro[i, batch[i][1]] = 1
            bce_target = np.concatenate((bce_target_dis, bce_target_pro), axis=1)

            # multi label margin loss
            # 修复：将硬编码的 1 改为 batch_size
            multi_target_dis = np.full((batch_size, voc_size[0]), -1)
            multi_target_pro = np.full((batch_size, voc_size[1]), -1)
            for i in range(batch_size):
                multi_target_dis[i, 0:len(batch[i][0])] = batch[i][0]
                multi_target_pro[i, 0:len(batch[i][1])] = batch[i][1]
            multi_target = np.concatenate((multi_target_dis, multi_target_pro), axis=1)

            loss_bce = F.binary_cross_entropy_with_logits(result, torch.tensor(bce_target, device=device))
            loss_multi = F.multilabel_margin_loss(result, torch.tensor(multi_target, device=device))
            # 目前只用 BCE
            loss = loss_bce
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
        loss_train /= len(data_train)
        # validation
        loss_val, ja, prauc, avg_p, avg_r, avg_f1, avg_cnt = evaluator_mask(
            model, data_val, voc_size, epoch, device, mode='pretrain_mask')

        if ja > best_ja_mask:
            best_epoch_mask, best_ja_mask = epoch, ja
        logging.info(
            f'Training Loss_mask: {loss_train:.4f}, Validation Loss_mask: {loss_val:.4f}, '
            f'best_ja: {best_ja_mask:.4f} at epoch {best_epoch_mask}\n'
        )
        tensorboard_write_mask(writer, loss_train, loss_val, ja, prauc, epoch_mask)
    save_pretrained_model(model, save_dir)

def main_nsp(args, model, optimizer, writer, data, data_train, data_val, device, save_dir, log_save_id):
    epoch_nsp = 0
    best_epoch_nsp, best_prc_nsp = 0, 0
    EPOCH = args.pretrain_epochs
    # 用 tqdm 包裹 nsp 预训练 epoch 循环
    for epoch in tqdm(range(EPOCH), ncols=60, desc="pretrain_nsp_epoch", total=EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} -------------model_name={args.model_name}, logger={log_save_id}, mode=pretrain_nsp')

        model.train()
        epoch_nsp += 1
        loss_train = 0
        for batch in tqdm(data_train, ncols=60, desc="pretrain_nsp", total=len(data_train)):
            nsp_batch, nsp_target = nsp_batch_data(batch, data, neg_sample_rate=1)
            result = model(nsp_batch, mode='pretrain_nsp')
            target_tensor = torch.tensor(nsp_target, device=device, dtype=torch.float32)

            # 二分类 NSP，用 BCEWithLogitsLoss
            if result.dim() == 2 and result.size(1) == 1:
                result_for_loss = result.view(-1)
            else:
                result_for_loss = result

            loss = F.binary_cross_entropy_with_logits(result_for_loss, target_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
        loss_train /= len(data_train)
        # validation
        precision, loss_val = evaluator_nsp(model, data_val, data, epoch, device, mode='pretrain_nsp')
        loss_val /= len(data_val)
        if precision > best_prc_nsp:
            best_epoch_nsp, best_prc_nsp = epoch, precision
        logging.info(
            f'Epoch {epoch:03d}   prc: {precision:.4}, best_prc: {best_prc_nsp:.4f} at epoch {best_epoch_nsp}, '
            f'Training Loss_nsp: {loss_train:.4f}, Validation Loss_nsp: {loss_val:.4f}\n'
        )
        tensorboard_write_nsp(writer, loss_train, loss_val, precision, epoch_nsp)
    save_pretrained_model(model, save_dir)

def save_pretrained_model(model, save_dir):
    # save the pretrained model
    model_path = os.path.join(save_dir, 'saved.pretrained_model')
    torch.save(model.state_dict(), open(model_path, 'wb'))
    logging.info('Pretrained model saved to {}'.format(model_path))

def load_pretrained_model(model, model_path):
    # load the pretrained model
    model.load_state_dict(torch.load(open(model_path, 'rb')))
    logging.info('Pretrained model loaded from {}'.format(model_path))

def loss_func(voc_size, patient, results, device):
    loss_bce_lst = torch.Tensor().double().to(device)
    loss_multi_lst = torch.Tensor().double().to(device)
    for idx, adm in enumerate(patient):
        result = results[idx].unsqueeze(dim=0)
        loss_bce_target = np.zeros((1, voc_size[2]))  # (1, drug_num)
        loss_bce_target[:, adm[2]] = 1

        loss_multi_target = np.full((1, voc_size[2]), -1)
        loss_multi_target[0][0:len(adm[2])] = adm[2]

        loss_bce = F.binary_cross_entropy_with_logits(result, torch.tensor(loss_bce_target, device=device))
        loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.tensor(loss_multi_target, device=device))
        loss_bce_lst = torch.cat([loss_bce_lst, loss_bce.view(-1)])
        loss_multi_lst = torch.cat([loss_multi_lst, loss_multi.view(-1)])
    loss_bce_patient = loss_bce_lst.sum()
    loss_multi_patient = loss_multi_lst.sum()
    return loss_bce_patient, loss_multi_patient

# 下面三个函数不再使用tensorboard，而是直接print并通过logging写入日志文件
def tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                      loss_train_bce=0.0, loss_train_multi=0.0, loss_train_ddi=0.0, loss_train_all=0.0,
                      loss_val_bce=0.0, loss_val_multi=0.0, loss_val_ddi=0.0, loss_val_all=0.0):
    msg_train = (f"[Epoch {epoch}] Train - BCE: {loss_train_bce:.4f}, "
                 f"MULTI: {loss_train_multi:.4f}, DDI: {loss_train_ddi:.4f}, SUM: {loss_train_all:.4f}")
    msg_val = (f"[Epoch {epoch}] Val   - BCE: {loss_val_bce:.4f}, "
               f"MULTI: {loss_val_multi:.4f}, DDI: {loss_val_ddi:.4f}, SUM: {loss_val_all:.4f}")
    msg_metrics = (f"[Epoch {epoch}] Metrics - Jaccard: {ja:.4f}, PRAUC: {prauc:.4f}, "
                   f"DDI: {ddi_rate:.4f}, Med_count: {avg_med:.4f}")

    print(msg_train)
    print(msg_val)
    print(msg_metrics)

    logging.info(msg_train)
    logging.info(msg_val)
    logging.info(msg_metrics)

def tensorboard_write_mask(writer, loss_train, loss_val, ja, prauc, epoch):
    msg = (f"[Mask][Epoch {epoch}] Train Loss: {loss_train:.4f}, "
           f"Val Loss: {loss_val:.4f}, Jaccard: {ja:.4f}, PRAUC: {prauc:.4f}")
    print(msg)
    logging.info(msg)

def tensorboard_write_nsp(writer, loss_train, loss_val, precision, epoch):
    msg = (f"[NSP][Epoch {epoch}] Train Loss: {loss_train:.4f}, "
           f"Val Loss: {loss_val:.4f}, Precision: {precision:.4f}")
    print(msg)
    logging.info(msg)

if __name__ == '__main__':
    torch.manual_seed(1203)
    np.random_seed = 1203
    np.random.seed(1203)
    random.seed(1203)

    # 记录开始时间
    _start_time = time.time()

    args = get_args()
    main(args)

    # 记录结束时间并计算总运行时间
    _end_time = time.time()
    _elapsed = int(_end_time - _start_time)
    _hours = _elapsed // 3600
    _minutes = (_elapsed % 3600) // 60
    _seconds = _elapsed % 60

    total_time_str = f"{_hours:02d}-h-{_minutes:02d}-m-{_seconds:02d}-s"
    print(f"Total running time: {total_time_str}")
    logging.info(f"Total running time: {total_time_str}")