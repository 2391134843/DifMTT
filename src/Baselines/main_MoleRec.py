import os
import sys

# 获取脚本所在目录，正确设置路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _src_dir)
sys.path.insert(0, os.path.dirname(_src_dir))

import argparse
from copy import deepcopy
from collections import defaultdict
import dill
import logging
import math
import numpy as np
from tqdm import tqdm
import time
import torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss

from Baselines.MoleRec import MoleRecModel
from models.gnn import graph_batch_from_smile
from utils.util import buildPrjSmiles, create_log_id, logging_config, get_model_path, \
    multi_label_metric, ddi_rate_score, get_grouped_metrics, get_n_params

def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)


def parse_args():
    parser = argparse.ArgumentParser('Experiment For DrugRec')
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='MoleRec', help="model name")
    parser.add_argument('--early_stop', type=int, default=10, help='early stop after this many epochs without improvement')
    parser.add_argument('--single', action='store_true', help='single visit mode')

    parser.add_argument('-t', '--test', action='store_true', help="evaluating mode")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default=None, help='log dir prefix like "log0", for model test')
    parser.add_argument('--test_after_train', action='store_true', help='train the model and then run test with the best checkpoint')
    parser.add_argument('--cuda', type=int, default=5, help='which cuda')

    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument(
        '--dataset', type=str, default='mimic-iii',
        help='dataset name, mimic-iii or mimic-iv'
    )
    parser.add_argument(
        '--target_ddi', type=float, default=0.12,
        help='expected ddi for training'
    )
    parser.add_argument(
        '--coef', default=2.5, type=float,
        help='coefficient for DDI Loss Weight Annealing'
    )
    parser.add_argument(
        '--embedding', action='store_false',
        help='use embedding table for substructures' +
        'if it\'s not chosen, the substructure will be encoded by GNN'
    )
    parser.add_argument(
        '--epochs', default=50, type=int,
        help='the epochs for training'
    )

    args = parser.parse_args()
    return args


def eval_one_epoch(model, dataset, data_eval, voc_size, drug_data, ddi_adj_path, mode = 'Val', rec_results_path=None):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1, visit_weights = [[] for _ in range(7)]
    med_cnt, visit_cnt = 0, 0
    ja_visit = [[] for _ in range(5)]
    f1_visit = [[] for _ in range(5)]
    prauc_visit = [[] for _ in range(5)]
    smm_record_visit = [[] for _ in range(5)]

    rec_results = [] 

    for step, input_seq in tqdm(enumerate(data_eval), ncols=60, desc='Evaluating', total=len(data_eval)):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        visit_weights_patient = []
        if mode == "Test":
            all_diseases = []
            all_procedures = []
            all_medications = []

        for adm_idx, adm in enumerate(input_seq):
            if mode == "Test":
                diseases = adm[0]
                procedures = adm[1]
                medications = adm[2]
                all_diseases.append(diseases)
                all_procedures.append(procedures)
                all_medications.append(medications)

            output, _ = model(
                patient_data=input_seq[:adm_idx + 1],
                **drug_data
            )
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)
            visit_weights_patient.append(adm[3])

            output = torch.sigmoid(output).detach().cpu().numpy()[0]
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        if mode == "Test":
            records = [all_diseases, all_procedures, all_medications, y_pred_label, visit_weights_patient, [adm_ja]]
            rec_results.append(records)

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        visit_weights.append(np.max(visit_weights_patient))
        if mode == "Test":
            if len(input_seq) < 5:
                visit_idx = len(input_seq) - 1
            else:
                visit_idx = 4
            ja_visit[visit_idx].append(adm_ja)
            f1_visit[visit_idx].append(adm_avg_f1)
            prauc_visit[visit_idx].append(adm_prauc)
            smm_record_visit[visit_idx].append(y_pred_label)
    
    if mode == "Test":
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
        dill.dump(rec_results, open(rec_results_file, 'wb'))
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)
    get_grouped_metrics(ja, visit_weights)
    if mode == "Test":
        # 计算每个visit组的DDI
        ddi_visit = []
        for i in range(5):
            if len(smm_record_visit[i]) > 0:
                ddi_v = ddi_rate_score(smm_record_visit[i], path=ddi_adj_path)
            else:
                ddi_v = 0.0
            ddi_visit.append(ddi_v)
        
        visit_metrics = {
            'ja': ja_visit,
            'f1': f1_visit,
            'prauc': prauc_visit,
            'ddi': ddi_visit
        }
        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
            np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt, visit_metrics
    else:
        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
            np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


def Test(model, dataset, model_path, device, data_test, voc_size, drug_data, ddi_adj_path, rec_results_path):
    with open(model_path, 'rb') as Fin:
        model.load_state_dict(torch.load(Fin, map_location=device))
    model = model.to(device).eval()
    logging.info('--------------------Begin Testing--------------------')
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med, visit_metrics = \
        eval_one_epoch(model, dataset, data_test, voc_size, drug_data, ddi_adj_path, "Test", rec_results_path)
    
    # 输出每个visit分组的详细指标
    print("\n========== 按就诊次数分组的指标 ==========")
    logging.info("\n========== 按就诊次数分组的指标 ==========")
    for i in range(5):
        ja_list = visit_metrics['ja'][i]
        f1_list = visit_metrics['f1'][i]
        prauc_list = visit_metrics['prauc'][i]
        ddi_v = visit_metrics['ddi'][i]
        
        n_samples = len(ja_list)
        if n_samples > 0:
            ja_mean = np.mean(ja_list)
            f1_mean = np.mean(f1_list)
            prauc_mean = np.mean(prauc_list)
        else:
            ja_mean = f1_mean = prauc_mean = 0.0
        
        msg = f"{i+1}visit (n={n_samples:4d}): JA={ja_mean:.4f}, F1={f1_mean:.4f}, PRAUC={prauc_mean:.4f}, DDI={ddi_v:.4f}"
        print(msg)
        logging.info(msg)
    
    outstring = f'\nddi_rate: {ddi_rate:.4f}, \n' +\
                f'ja      : {ja:.4f}, \n' +\
                f'avg_f1  : {avg_f1:.4f}, \n' +\
                f'prauc   : {prauc:.4f}, \n' +\
                f'avg_p   : {avg_p:.4f}, \n' +\
                f'avg_r   : {avg_r:.4f}, \n' +\
                f'med     : {avg_med:.4f}, \n'
    print(outstring)
    logging.info(outstring)



def Train(
    model, dataset, device, data_train, data_eval, voc_size, drug_data,
    optimizer, save_dir, coef, target_ddi, ddi_adj_path, EPOCH=50
):
    history, best_epoch, best_ja = defaultdict(list), 0, 0
    best_model_state, best_ddi_rate, best_model_path = None, 0, None
    total_train_time, ddi_losses, ddi_values = 0, [], []
    for epoch in range(EPOCH):
        logging.info(f'----------------Epoch {epoch + 1}------------------')
        model = model.train()
        tic, ddi_losses_epoch = time.time(), []
        for step, input_seq in tqdm(enumerate(data_train), ncols=60, desc='Training', total=len(data_train)):
            for adm_idx, adm in enumerate(input_seq):
                bce_target = torch.zeros((1, voc_size[2])).to(device)
                bce_target[:, adm[2]] = 1

                multi_target = -torch.ones((1, voc_size[2])).long()
                for idx, item in enumerate(adm[2]):
                    multi_target[0][idx] = item
                multi_target = multi_target.to(device)

                result, loss_ddi = model(
                    patient_data=input_seq[:adm_idx + 1],
                    **drug_data
                )

                sigmoid_res = torch.sigmoid(result)

                loss_bce = binary_cross_entropy_with_logits(result, bce_target)
                loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)

                result = sigmoid_res.detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], path=ddi_adj_path
                )

                if current_ddi_rate <= target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = coef * (1 - (current_ddi_rate / target_ddi))
                    beta = min(math.exp(beta), 1)
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) \
                        + (1 - beta) * loss_ddi

                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ddi_losses.append(sum(ddi_losses_epoch) / len(ddi_losses_epoch))
        logging.info(f'\nddi_loss : {ddi_losses[-1]}\n')
        train_time, tic = time.time() - tic, time.time()
        total_train_time += train_time
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_one_epoch(model, dataset, data_eval, voc_size, drug_data, ddi_adj_path)
        logging.info(f'training time: {train_time}, testing time: {time.time() - tic}')
        ddi_values.append(ddi_rate)
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            logging.info('ddi: {:.4f}, Med: {:.4f}, Ja: {:.4f}, F1: {:.4f}, Pre: {:.4f}, Rec: {:.4f},  PRAUC: {:.4f}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['avg_p'][-5:]),
                np.mean(history['avg_r'][-5:]),
                np.mean(history['prauc'][-5:])
            ))

        model_name = 'Epoch_{}_TARGET_{:.2f}_JA_{:.4f}_DDI_{:.4f}.model'.format(
            epoch, target_ddi, ja, ddi_rate
        )
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_ddi_rate = ja, ddi_rate
            best_model_state = deepcopy(model.state_dict()) 
        logging.info(f'best_epoch: {best_epoch}, best_ja: {best_ja:.4f}\n')

        if epoch - best_epoch > args.early_stop:   # n个epoch内，验证集性能不上升之后就停
            break

    logging.info('avg training time/epoch: {:.4f}'.format(total_train_time / EPOCH))
    if best_model_state is None:
        best_model_state = deepcopy(model.state_dict())
        best_ddi_rate = ddi_values[-1] if len(ddi_values) > 0 else 0
    best_model_path = os.path.join(save_dir, 'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, best_ddi_rate))
    torch.save(best_model_state, open(best_model_path, 'wb'))
    return best_model_path, best_epoch, best_ja, best_ddi_rate



if __name__ == '__main__':
    set_seed()
    args = parse_args()
    logging.info(args)

    # set logger
    if args.test:
        args.note = 'test of ' + args.log_dir_prefix
    _log_base = os.path.join(_src_dir, 'log')
    log_directory_path = os.path.join(_log_base, args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    if not torch.cuda.is_available() or args.cuda < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.cuda}')

    dataset = args.dataset
    _data_dir = os.path.join(os.path.dirname(_src_dir), 'data', 'output', dataset)
    data_path = os.path.join(_data_dir, 'records_final.pkl')
    voc_path = os.path.join(_data_dir, 'voc_final.pkl')
    ddi_adj_path = os.path.join(_data_dir, 'ddi_A_final.pkl')
    ddi_mask_path = os.path.join(_data_dir, 'ddi_mask_H.pkl')
    molecule_path = os.path.join(_data_dir, 'atc3toSMILES.pkl')
    substruct_smile_path = os.path.join(_data_dir, 'substructure_smiles.pkl')

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)

    diag_voc, pro_voc, med_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point:split_point + eval_len]
    data_test = data[split_point + eval_len:]

    if args.single:
        # convert data into single visit format
        data_train = [[visit] for patient in data_train for visit in patient]
        data_eval = [[visit] for patient in data_eval for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]

    average_projection, smiles_list = \
        buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)

    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': 4, 'emb_dim': args.dim, 'graph_pooling': 'mean',
        'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False
    }

    if args.embedding:
        substruct_para, substruct_forward = None, None
    else:
        with open(substruct_smile_path, 'rb') as Fin:
            substruct_smiles_list = dill.load(Fin)

        substruct_graphs = graph_batch_from_smile(substruct_smiles_list)
        substruct_forward = {'batched_data': substruct_graphs.to(device)}
        substruct_para = {
            'num_layer': 4, 'emb_dim': args.dim, 'graph_pooling': 'mean',
            'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False
        }

    model = MoleRecModel(
        global_para=molecule_para, substruct_para=substruct_para,
        emb_dim=args.dim, global_dim=args.dim, substruct_dim=args.dim,
        substruct_num=ddi_mask_H.shape[1], voc_size=voc_size,
        use_embedding=args.embedding, device=device, dropout=args.dp
    ).to(device)

    drug_data = {
        'substruct_data': substruct_forward,
        'mol_data': molecule_forward,
        'ddi_mask_H': ddi_mask_H,
        'tensor_ddi_adj': ddi_adj,
        'average_projection': average_projection
    }

    if args.test:
        rec_results_path = save_dir + '/' + 'rec_results'

        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        Test(model, dataset, model_path, device, data_test, voc_size, drug_data, ddi_adj_path, rec_results_path)
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        optimizer = Adam(model.parameters(), lr=args.lr)
        best_model_path, best_epoch, best_ja, best_ddi_rate = Train(
            model, dataset, device, data_train, data_eval, voc_size, drug_data,
            optimizer, save_dir, args.coef, args.target_ddi, ddi_adj_path, EPOCH=args.epochs
        )
        if args.test_after_train:
            rec_results_path = save_dir + '/' + 'rec_results'
            Test(model, dataset, best_model_path, device, data_test, voc_size, drug_data, ddi_adj_path, rec_results_path)
