import argparse
from copy import deepcopy
from collections import defaultdict
import dill
import logging
import math
import numpy as np
import os
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
import sys

from models.DifMTT import DifMTTModel
from models.gnn import graph_batch_from_smile
from utils.util import buildPrjSmiles, create_log_id, logging_config, get_model_path, \
    multi_label_metric, ddi_rate_score, get_grouped_metrics, get_n_params


def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)


def _fmt_array_4(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return "[]"
    return np.array2string(
        arr.astype(np.float64),
        formatter={'float_kind': lambda v: f"{v:.4f}"},
        separator=', '
    )


def _print_array(name, x):
    arr = np.asarray(x)
    print(f"[{name}] shape={arr.shape} values={_fmt_array_4(arr)}")


def _format_hms(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}-H-{m:02d}-M-{s:02d}-S"
# -------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser('Experiment For DrugRec')
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='DifMTT', help="model name")
    parser.add_argument('--early_stop', type=int, default=15, help='early stop after this many epochs without improvement')
    parser.add_argument('--single', action='store_true', help='single visit mode')

    parser.add_argument('-t', '--test', action='store_true', help="evaluating mode")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default=None, help='log dir prefix like "log0", for model test')
    parser.add_argument('--test_after_train', action='store_true', help='train the model and then run test with the best checkpoint')
    parser.add_argument('--cuda', type=int, default=5, help='which cuda')

    parser.add_argument('--dim', default=256, type=int, help='model dimension')
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
    
    parser.add_argument('--sbcl', action='store_true', help='enable SBCL auxiliary training')
    parser.add_argument('--sbcl_weight', type=float, default=0.1, help='weight of SBCL loss')
    parser.add_argument('--sbcl_warmup_epochs', type=int, default=5, help='warm-up epochs before SBCL starts')
    parser.add_argument('--sbcl_update_interval', type=int, default=2, help='update clustering & temperature every K epochs')
    parser.add_argument('--sbcl_delta', type=int, default=50, help='delta in SBCL clustering threshold M=max(n_C, delta)')
    parser.add_argument('--sbcl_cluster_iter', type=int, default=1, help='iterations K in balanced clustering (Algorithm 1)')
    parser.add_argument('--sbcl_tau1', type=float, default=0.07, help='temperature tau1 for subclass-level loss')
    parser.add_argument('--sbcl_alpha', type=float, default=1.0, help='alpha in dynamic temperature computation')
    parser.add_argument('--sbcl_beta', type=float, default=1.0, help='weight for class-level contrastive term')
    parser.add_argument('--sbcl_num_pos', type=int, default=20, help='max number of positives sampled per anchor')
    parser.add_argument('--sbcl_num_neg', type=int, default=256, help='number of negatives sampled per anchor')
    parser.add_argument('--multi_intent', action='store_true', help='enable multi-intent learning')
    parser.add_argument('--num_intents', type=int, default=4, help='number of treatment intents K')
    parser.add_argument('--intent_weight', type=float, default=0.2, help='weight of intent contrastive loss')
    parser.add_argument('--intent_warmup_epochs', type=int, default=3, help='warm-up epochs before intent CL starts')
    parser.add_argument('--intent_update_interval', type=int, default=2, help='update intent prototypes every K epochs')
    parser.add_argument('--intent_tau', type=float, default=0.1, help='temperature for intent contrastive loss')
    parser.add_argument('--intent_num_neg', type=int, default=128, help='number of negatives for intent CL')
    parser.add_argument('--diffusion', action='store_true', help='enable diffusion auxiliary modeling (DiffuRec-style)')
    parser.add_argument('--diff_steps', type=int, default=50, help='diffusion steps T')
    parser.add_argument('--diff_beta_start', type=float, default=1e-4, help='beta schedule start')
    parser.add_argument('--diff_beta_end', type=float, default=0.02, help='beta schedule end')
    parser.add_argument('--diff_weight', type=float, default=0.1, help='weight of diffusion loss')
    parser.add_argument('--diff_warmup_epochs', type=int, default=5, help='warm-up epochs before diffusion loss starts')
    parser.add_argument('--diff_ddi_weight', type=float, default=0.0, help='DDI regularization weight inside diffusion loss')
    parser.add_argument('--diff_infer', action='store_true', help='use diffusion sampling at inference/eval (fused with original score)')
    parser.add_argument('--diff_guidance_ddi', type=float, default=0.0, help='DDI guidance strength during diffusion sampling')
    parser.add_argument('--diff_fuse_alpha', type=float, default=0.5, help='fuse score = (1-a)*orig + a*diff')
    parser.add_argument('--diff_sample_steps', type=int, default=None, help='use fewer reverse steps at sampling (<=diff_steps)')
    parser.add_argument('--mol_num_layer', type=int, default=4, help='num layers of molecule GNN')
    parser.add_argument('--mol_emb_dim', type=int, default=None, help='embedding dim of molecule GNN (default: --dim)')
    parser.add_argument('--mol_graph_pooling', type=str, default='mean', help='graph pooling for molecule GNN')
    parser.add_argument('--mol_drop_ratio', type=float, default=None, help='drop ratio of molecule GNN (default: --dp)')
    parser.add_argument('--mol_gnn_type', type=str, default='gin', help='gnn type for molecule encoder')
    parser.add_argument('--mol_virtual_node', action='store_true', help='use virtual node for molecule GNN')
    parser.add_argument('--sub_num_layer', type=int, default=4, help='num layers of substructure GNN')
    parser.add_argument('--sub_emb_dim', type=int, default=None, help='embedding dim of substructure GNN (default: --dim)')
    parser.add_argument('--sub_graph_pooling', type=str, default='mean', help='graph pooling for substructure GNN')
    parser.add_argument('--sub_drop_ratio', type=float, default=None, help='drop ratio of substructure GNN (default: --dp)')
    parser.add_argument('--sub_gnn_type', type=str, default='gin', help='gnn type for substructure encoder')
    parser.add_argument('--sub_virtual_node', action='store_true', help='use virtual node for substructure GNN')

    args = parser.parse_args()

    if args.mol_emb_dim is None:
        args.mol_emb_dim = args.dim
    if args.mol_drop_ratio is None:
        args.mol_drop_ratio = args.dp
    if args.sub_emb_dim is None:
        args.sub_emb_dim = args.dim
    if args.sub_drop_ratio is None:
        args.sub_drop_ratio = args.dp

    return args


def eval_one_epoch(model, dataset, data_eval, voc_size, drug_data, mode='Val', rec_results_path=None, args=None):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1, visit_weights = [[] for _ in range(7)]
    med_cnt, visit_cnt = 0, 0
    ja_visit = [[] for _ in range(5)]
    f1_visit = [[] for _ in range(5)]
    prauc_visit = [[] for _ in range(5)]
    smm_record_visit = [[] for _ in range(5)]  # 用于计算每组DDI

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
                diffusion_infer=bool(getattr(args, 'diff_infer', False)) if args is not None else False,
                diffusion_guidance_scale=float(getattr(args, 'diff_guidance_ddi', 0.0)) if args is not None else 0.0,
                diffusion_fuse_alpha=float(getattr(args, 'diff_fuse_alpha', 0.5)) if args is not None else 0.5,
                diffusion_sample_steps=getattr(args, 'diff_sample_steps', None) if args is not None else None,
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

    ddi_rate = ddi_rate_score(smm_record, path=f'../data/output/{dataset}/ddi_A_final.pkl')
    get_grouped_metrics(ja, visit_weights)

    if mode == "Test":
        # 计算每个visit组的DDI
        ddi_visit = []
        for i in range(5):
            if len(smm_record_visit[i]) > 0:
                ddi_v = ddi_rate_score(smm_record_visit[i], path=f'../data/output/{dataset}/ddi_A_final.pkl')
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


def Test(model, dataset, model_path, device, data_test, voc_size, drug_data, rec_results_path, args=None):
    with open(model_path, 'rb') as Fin:
        model.load_state_dict(torch.load(Fin, map_location=device))
    model = model.to(device).eval()

    print('''
################################################################################
                              BEGIN TESTING
################################################################################''')

    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med, visit_metrics = \
        eval_one_epoch(model, dataset, data_test, voc_size, drug_data, "Test", rec_results_path, args=args)

    print('''
--------------------------------------------------------------------------------
  Metrics by Visit Count
--------------------------------------------------------------------------------''')
    print(f"  {'Visit':<12} {'Samples':>8} {'Jaccard':>10} {'F1':>10} {'PRAUC':>10} {'DDI':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
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
        
        visit_label = f"{i+1} visit" if i < 4 else "5+ visits"
        print(f"  {visit_label:<12} {n_samples:>8} {ja_mean:>10.4f} {f1_mean:>10.4f} {prauc_mean:>10.4f} {ddi_v:>10.4f}")

    print(f'''
--------------------------------------------------------------------------------
  Overall Test Results
--------------------------------------------------------------------------------
    DDI Rate           : {ddi_rate:.4f}
    Jaccard            : {ja:.4f}
    F1 Score           : {avg_f1:.4f}
    Precision          : {avg_p:.4f}
    Recall             : {avg_r:.4f}
    PRAUC              : {prauc:.4f}
    Avg Med Count      : {avg_med:.4f}
################################################################################
''')



class SBCLState:
    def __init__(self, enabled: bool, diag_voc_size: int, device: torch.device, args):
        self.enabled = enabled
        self.diag_voc_size = diag_voc_size
        self.device = device
        self.args = args

        self.ready = False
        self.bank_z = None
        self.bank_class = None
        self.bank_subclass = None

        self.class_labels = None
        self.subclass_labels = None
        self.sample_index_map = {}

        self.cls_to_indices = {}
        self.sub_to_indices = {}
        self.tau2 = {}

    @staticmethod
    def _primary_diag(adm):
        if adm is None or len(adm) == 0:
            return -1
        diags = adm[0]
        if diags is None or len(diags) == 0:
            return -1
        return int(diags[0])

    @staticmethod
    def _farthest_point_init(features: np.ndarray, m: int) -> np.ndarray:
        n = features.shape[0]
        if m <= 1:
            return np.array([0], dtype=np.int64)
        rng = np.random.default_rng(2048)
        centers = [int(rng.integers(0, n))]
        for _ in range(1, m):
            sims = features @ features[np.array(centers)].T
            max_sim = sims.max(axis=1)
            next_idx = int(np.argmin(max_sim))
            centers.append(next_idx)
        return np.array(centers, dtype=np.int64)

    @staticmethod
    def _balanced_clustering(features: np.ndarray, m: int, M: int, iters: int = 1) -> np.ndarray:
        n, d = features.shape
        if m <= 1 or n <= 1:
            return np.zeros(n, dtype=np.int64)

        centers_idx = SBCLState._farthest_point_init(features, m)
        centers = features[centers_idx].copy()

        for _ in range(max(iters, 1)):
            sim = features @ centers.T
            flat = sim.reshape(-1)
            order = np.argsort(-flat)
            assigned = -np.ones(n, dtype=np.int64)
            cap = np.zeros(m, dtype=np.int64)

            for idx in order:
                i = idx // m
                j = idx % m
                if assigned[i] != -1:
                    continue
                if cap[j] >= M:
                    continue
                assigned[i] = j
                cap[j] += 1
                if (assigned != -1).all():
                    break

            if (assigned == -1).any():
                remaining = np.where(assigned == -1)[0]
                for i in remaining:
                    valid = np.where(cap < M)[0]
                    if len(valid) == 0:
                        valid = np.arange(m)
                    j = valid[np.argmax(sim[i, valid])]
                    assigned[i] = int(j)
                    cap[j] += 1

            new_centers = centers.copy()
            for j in range(m):
                members = features[assigned == j]
                if len(members) > 0:
                    c = members.mean(axis=0)
                    c = c / (np.linalg.norm(c) + 1e-12)
                    new_centers[j] = c
            centers = new_centers

        return assigned

    def update(self, model, data_train, epoch: int):
        if not self.enabled:
            return

        was_training = model.training
        model.eval()

        emb_list = []
        cls_list = []
        index_map = {}

        with torch.no_grad():
            global_idx = 0
            for patient_idx, input_seq in enumerate(tqdm(data_train, ncols=60, desc='SBCL bank', total=len(data_train))):
                for adm_idx, adm in enumerate(input_seq):
                    c = self._primary_diag(adm)
                    if c < 0:
                        continue
                    z = model.encode_patient(input_seq[:adm_idx + 1], normalize=True)
                    emb_list.append(z.detach().cpu())
                    cls_list.append(c)
                    index_map[(patient_idx, adm_idx)] = global_idx
                    global_idx += 1

        if was_training:
            model.train()

        if len(emb_list) == 0:
            self.ready = False
            return

        Z = torch.stack(emb_list, dim=0).numpy().astype(np.float32)
        class_labels = np.array(cls_list, dtype=np.int64)
        N = Z.shape[0]

        unique, counts = np.unique(class_labels, return_counts=True)
        n_C = int(counts.min()) if len(counts) > 0 else 0
        M = max(n_C, int(self.args.sbcl_delta))

        subclass_labels = -np.ones(N, dtype=np.int64)
        next_sub_id = 0

        for c, n_c in zip(unique.tolist(), counts.tolist()):
            idxs = np.where(class_labels == c)[0]
            if n_c <= M or n_c <= 1:
                subclass_labels[idxs] = next_sub_id
                next_sub_id += 1
                continue

            m = int(math.ceil(n_c / M))
            assign = self._balanced_clustering(
                features=Z[idxs], m=m, M=M,
                iters=int(self.args.sbcl_cluster_iter)
            )
            for j in range(m):
                subclass_labels[idxs[assign == j]] = next_sub_id + j
            next_sub_id += m

        alpha = float(self.args.sbcl_alpha)
        tau1 = float(self.args.sbcl_tau1)

        phi = {}
        for c in unique.tolist():
            idxs = np.where(class_labels == c)[0]
            if len(idxs) <= 1:
                phi[c] = 0.0
                continue
            tc = Z[idxs].mean(axis=0)
            tc = tc / (np.linalg.norm(tc) + 1e-12)
            dists = np.linalg.norm(Z[idxs] - tc[None, :], axis=1)
            denom = (len(idxs) * math.log(len(idxs) + alpha))
            phi[c] = float(dists.sum() / (denom + 1e-12))

        mean_phi = float(np.mean(list(phi.values()))) if len(phi) > 0 else 1.0
        tau2 = {}
        for c in unique.tolist():
            tau2[c] = float(tau1 * math.exp(phi[c] / (mean_phi + 1e-12)))

        cls_to_indices = {}
        sub_to_indices = {}
        for i, (c, s) in enumerate(zip(class_labels.tolist(), subclass_labels.tolist())):
            cls_to_indices.setdefault(c, []).append(i)
            sub_to_indices.setdefault(s, []).append(i)

        cls_to_indices = {k: np.array(v, dtype=np.int64) for k, v in cls_to_indices.items()}
        sub_to_indices = {k: np.array(v, dtype=np.int64) for k, v in sub_to_indices.items()}

        self.bank_z = torch.tensor(Z, device=self.device)
        self.bank_class = torch.tensor(class_labels, device=self.device)
        self.bank_subclass = torch.tensor(subclass_labels, device=self.device)

        self.class_labels = class_labels
        self.subclass_labels = subclass_labels
        self.sample_index_map = index_map
        self.cls_to_indices = cls_to_indices
        self.sub_to_indices = sub_to_indices
        self.tau2 = tau2
        self.ready = True

        logging.info(f'[SBCL] epoch={epoch} | N={N} | tail_nC={n_C} | M={M} | subclasses={next_sub_id}')

    def _sample_indices_reject(self, N: int, need: int, reject_mask: np.ndarray) -> np.ndarray:
        if need <= 0:
            return np.array([], dtype=np.int64)
        out = []
        rng = np.random.default_rng()
        while len(out) < need:
            cand = int(rng.integers(0, N))
            if reject_mask[cand]:
                continue
            out.append(cand)
        return np.array(out, dtype=np.int64)

    def bi_granularity_loss(self, z_anchor: torch.Tensor, class_id: int, subclass_id: int) -> torch.Tensor:
        if not self.ready:
            return torch.zeros((), device=self.device)

        pos_sub = self.sub_to_indices.get(subclass_id, np.array([], dtype=np.int64))
        pos_cls = self.cls_to_indices.get(class_id, np.array([], dtype=np.int64))

        if len(pos_sub) > 0 and len(pos_cls) > 0:
            pos_cls = pos_cls[~np.isin(pos_cls, pos_sub)]

        rng = np.random.default_rng()
        if len(pos_sub) > self.args.sbcl_num_pos:
            pos_sub = rng.choice(pos_sub, size=self.args.sbcl_num_pos, replace=False)
        if len(pos_cls) > self.args.sbcl_num_pos:
            pos_cls = rng.choice(pos_cls, size=self.args.sbcl_num_pos, replace=False)

        N = int(self.bank_z.shape[0])
        num_neg = int(self.args.sbcl_num_neg)

        reject_sub = (self.subclass_labels == subclass_id)
        neg_sub = self._sample_indices_reject(N, num_neg, reject_sub)

        reject_cls = (self.class_labels == class_id)
        neg_cls = self._sample_indices_reject(N, num_neg, reject_cls)

        def info_nce(pos_idx: np.ndarray, neg_idx: np.ndarray, tau: float) -> torch.Tensor:
            if len(pos_idx) == 0:
                return torch.zeros((), device=self.device)
            cand_idx = np.concatenate([pos_idx, neg_idx], axis=0)
            cand = self.bank_z.index_select(0, torch.tensor(cand_idx, device=self.device))
            sims = torch.matmul(cand, z_anchor)
            logits = sims / tau
            denom = torch.logsumexp(logits, dim=0)
            pos_logits = logits[:len(pos_idx)]
            return (-(pos_logits - denom)).mean()

        tau1 = float(self.args.sbcl_tau1)
        tau2 = float(self.tau2.get(class_id, tau1 * 2.0))

        loss_sub = info_nce(pos_sub, neg_sub, tau1)
        loss_cls = info_nce(pos_cls, neg_cls, tau2)

        loss = loss_sub + float(self.args.sbcl_beta) * loss_cls
        return loss



class IntentState:

    def __init__(self, enabled: bool, num_intents: int, intent_dim: int, 
                 device: torch.device, args):
        self.enabled = enabled
        self.num_intents = num_intents
        self.intent_dim = intent_dim
        self.device = device
        self.args = args
        
        self.ready = False
        self.intent_prototypes = None
        self.sample_to_intent = {}
        self.intent_to_samples = {}
        self.bank_z = None
        self.sample_index_map = {}
    
    def update(self, model, data_train, epoch: int):

        if not self.enabled:
            return
        
        was_training = model.training
        model.eval()
        
        emb_list = []
        index_map = {}
        
        with torch.no_grad():
            global_idx = 0
            for patient_idx, input_seq in enumerate(tqdm(data_train, ncols=60, desc='Intent bank', total=len(data_train))):
                for adm_idx, adm in enumerate(input_seq):
                    agg_intent, _, _ = model.encode_patient_intents(
                        input_seq[:adm_idx + 1], normalize=True
                    )
                    emb_list.append(agg_intent.detach().cpu())
                    index_map[(patient_idx, adm_idx)] = global_idx
                    global_idx += 1
        
        if was_training:
            model.train()
        
        if len(emb_list) == 0:
            self.ready = False
            return
        
        Z = torch.stack(emb_list, dim=0).numpy().astype(np.float32)  # (N, intent_dim)
        N = Z.shape[0]
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_intents, random_state=2048, n_init=10)
        cluster_labels = kmeans.fit_predict(Z)
        prototypes = kmeans.cluster_centers_  # (K, intent_dim)
        
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12)
        
        sample_to_intent = {}
        intent_to_samples = {k: [] for k in range(self.num_intents)}
        
        for i, label in enumerate(cluster_labels):
            sample_to_intent[i] = int(label)
            intent_to_samples[int(label)].append(i)
        
        intent_to_samples = {k: np.array(v, dtype=np.int64) for k, v in intent_to_samples.items()}
        
        self.intent_prototypes = torch.tensor(prototypes, dtype=torch.float32, device=self.device)
        self.bank_z = torch.tensor(Z, device=self.device)
        self.sample_to_intent = sample_to_intent
        self.intent_to_samples = intent_to_samples
        self.sample_index_map = index_map
        self.ready = True
        
        logging.info(f'[Intent] epoch={epoch} | N={N} | K={self.num_intents} | cluster_sizes={[len(v) for v in intent_to_samples.values()]}')
    
    def intent_contrastive_loss(self, z_anchor: torch.Tensor, patient_idx: int, adm_idx: int) -> torch.Tensor:

        if not self.ready:
            return torch.zeros((), device=self.device)
        
        global_idx = self.sample_index_map.get((patient_idx, adm_idx), None)
        if global_idx is None:
            return torch.zeros((), device=self.device)
        
        intent_id = self.sample_to_intent.get(global_idx, None)
        if intent_id is None:
            return torch.zeros((), device=self.device)
        
        pos_proto = self.intent_prototypes[intent_id]  # (intent_dim,)
        
        neg_proto_indices = [i for i in range(self.num_intents) if i != intent_id]
        
        tau = float(self.args.intent_tau)
        
        pos_sim = torch.dot(z_anchor, pos_proto) / tau
        
        all_sims = torch.matmul(self.intent_prototypes, z_anchor) / tau  # (K,)
        
        loss = -pos_sim + torch.logsumexp(all_sims, dim=0)
        
        return loss
    
    def intent_contrastive_loss_with_fnm(self, z_anchor: torch.Tensor, patient_idx: int, 
                                          adm_idx: int, batch_intents: list) -> torch.Tensor:

        if not self.ready:
            return torch.zeros((), device=self.device)
        
        global_idx = self.sample_index_map.get((patient_idx, adm_idx), None)
        if global_idx is None:
            return torch.zeros((), device=self.device)
        
        intent_id = self.sample_to_intent.get(global_idx, None)
        if intent_id is None:
            return torch.zeros((), device=self.device)
        
        tau = float(self.args.intent_tau)
        
        pos_proto = self.intent_prototypes[intent_id]
        pos_sim = torch.dot(z_anchor, pos_proto) / tau
        
        neg_mask = torch.ones(self.num_intents, dtype=torch.bool, device=self.device)
        neg_mask[intent_id] = False
        
        for bid in batch_intents:
            if bid is not None and bid != intent_id:
                pass
        
        neg_protos = self.intent_prototypes[neg_mask]  # (K-1, intent_dim)
        neg_sims = torch.matmul(neg_protos, z_anchor) / tau  # (K-1,)
        
        # InfoNCE
        all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims], dim=0)
        loss = -pos_sim + torch.logsumexp(all_sims, dim=0)
        
        return loss



def Train(
    model, dataset, device, data_train, data_eval, voc_size, drug_data,
    optimizer, save_dir, coef, target_ddi, EPOCH=50,
    sbcl_state=None, intent_state=None, args=None
):
    history, best_epoch, best_ja = defaultdict(list), 0, 0
    best_model_state, best_ddi_rate, best_model_path = None, 0, None
    total_train_time, ddi_losses, ddi_values = 0, [], []
    
    for epoch in range(EPOCH):
        logging.info(f'''
================================================================================
                              EPOCH {epoch + 1:3d} / {EPOCH}
================================================================================''')
        
        # SBCL update
        if sbcl_state is not None and sbcl_state.enabled and epoch >= args.sbcl_warmup_epochs and \
                (epoch - args.sbcl_warmup_epochs) % max(args.sbcl_update_interval, 1) == 0:
            sbcl_state.update(model, data_train, epoch)
        
        # Intent prototype update (E-step)
        if intent_state is not None and intent_state.enabled and epoch >= args.intent_warmup_epochs and \
                (epoch - args.intent_warmup_epochs) % max(args.intent_update_interval, 1) == 0:
            intent_state.update(model, data_train, epoch)
        
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

                use_intent_cl = (intent_state is not None and intent_state.enabled and 
                                epoch >= args.intent_warmup_epochs and intent_state.ready)
                use_sbcl = (sbcl_state is not None and sbcl_state.enabled and 
                           epoch >= args.sbcl_warmup_epochs and sbcl_state.ready)
                use_diffusion = (bool(getattr(args, 'diffusion', False)) and getattr(model, 'use_diffusion', False) and epoch >= int(getattr(args, 'diff_warmup_epochs', 0)))

                
                if use_intent_cl and model.use_multi_intent:
                    if use_diffusion:
                        result, loss_ddi, diff_loss, agg_intent, intent_reprs, intent_weights = model(
                            patient_data=input_seq[:adm_idx + 1],
                            return_intent_info=True,
                            return_diffusion_loss=True,
                            y_true=bce_target,
                            **drug_data
                        )
                    else:
                        result, loss_ddi, agg_intent, intent_reprs, intent_weights = model(
                            patient_data=input_seq[:adm_idx + 1],
                            return_intent_info=True,
                            **drug_data
                        )
                    z_anchor = agg_intent
                elif use_sbcl or use_intent_cl:
                    if use_diffusion:
                        result, loss_ddi, diff_loss, z_anchor = model(
                            patient_data=input_seq[:adm_idx + 1],
                            return_query=True,
                            return_diffusion_loss=True,
                            y_true=bce_target,
                            **drug_data
                        )
                    else:
                        result, loss_ddi, z_anchor = model(
                            patient_data=input_seq[:adm_idx + 1],
                            return_query=True,
                            **drug_data
                        )
                else:
                    if use_diffusion:
                        result, loss_ddi, diff_loss = model(
                            patient_data=input_seq[:adm_idx + 1],
                            return_diffusion_loss=True,
                            y_true=bce_target,
                            **drug_data
                        )
                    else:
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
                    [[y_label]], path=f'../data/output/{dataset}/ddi_A_final.pkl'
                )

                if current_ddi_rate <= target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = coef * (1 - (current_ddi_rate / target_ddi))
                    beta = min(math.exp(beta), 1)
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) \
                        + (1 - beta) * loss_ddi

                # SBCL auxiliary loss
                if use_sbcl:
                    sbcl_idx = sbcl_state.sample_index_map.get((step, adm_idx), None)
                    if sbcl_idx is not None:
                        c_id = int(sbcl_state.class_labels[sbcl_idx])
                        s_id = int(sbcl_state.subclass_labels[sbcl_idx])
                        sbcl_loss = sbcl_state.bi_granularity_loss(z_anchor, c_id, s_id)
                        loss = loss + float(args.sbcl_weight) * sbcl_loss

                # Intent contrastive loss (M-step)
                if use_intent_cl:
                    intent_loss = intent_state.intent_contrastive_loss(z_anchor, step, adm_idx)
                    loss = loss + float(args.intent_weight) * intent_loss

                # Diffusion auxiliary loss
                if use_diffusion:
                    loss = loss + float(args.diff_weight) * diff_loss

                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ddi_losses.append(sum(ddi_losses_epoch) / len(ddi_losses_epoch))
        train_time, tic = time.time() - tic, time.time()
        total_train_time += train_time
        
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_one_epoch(model, dataset, data_eval, voc_size, drug_data, args=args)
        eval_time = time.time() - tic
        
        logging.info(f'''
--------------------------------------------------------------------------------
  Training Summary
--------------------------------------------------------------------------------
    DDI Loss (avg)     : {ddi_losses[-1]:.6f}
    Training Time      : {train_time:.2f}s
    Evaluation Time    : {eval_time:.2f}s
--------------------------------------------------------------------------------''')
        
        ddi_values.append(ddi_rate)
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        logging.info(f'''
  Validation Metrics
--------------------------------------------------------------------------------
    DDI Rate           : {ddi_rate:.4f}
    Jaccard            : {ja:.4f}
    F1 Score           : {avg_f1:.4f}
    Precision          : {avg_p:.4f}
    Recall             : {avg_r:.4f}
    PRAUC              : {prauc:.4f}
    Avg Med Count      : {avg_med:.4f}
--------------------------------------------------------------------------------''')

        if epoch >= 5:
            logging.info(f'''
  Moving Average (Last 5 Epochs)
--------------------------------------------------------------------------------
    DDI Rate           : {np.mean(history['ddi_rate'][-5:]):.4f}
    Jaccard            : {np.mean(history['ja'][-5:]):.4f}
    F1 Score           : {np.mean(history['avg_f1'][-5:]):.4f}
    Precision          : {np.mean(history['avg_p'][-5:]):.4f}
    Recall             : {np.mean(history['avg_r'][-5:]):.4f}
    PRAUC              : {np.mean(history['prauc'][-5:]):.4f}
    Avg Med Count      : {np.mean(history['med'][-5:]):.4f}
--------------------------------------------------------------------------------''')

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_ddi_rate = ja, ddi_rate
            best_model_state = deepcopy(model.state_dict())
            logging.info(f'  [NEW BEST] Epoch {best_epoch} achieved best Jaccard: {best_ja:.4f}')
        logging.info(f'''
  Current Best
--------------------------------------------------------------------------------
    Best Epoch         : {best_epoch}
    Best Jaccard       : {best_ja:.4f}
    Early Stop Counter : {epoch - best_epoch} / {args.early_stop}
================================================================================
''')

        if epoch - best_epoch > args.early_stop:
            break

    logging.info(f'''
################################################################################
                           TRAINING COMPLETED
################################################################################
    Total Epochs       : {epoch + 1}
    Best Epoch         : {best_epoch}
    Best Jaccard       : {best_ja:.4f}
    Best DDI Rate      : {best_ddi_rate:.4f}
    Avg Time per Epoch : {total_train_time / (epoch + 1):.2f}s
    Total Training Time: {_format_hms(total_train_time)}
################################################################################
''')
    if best_model_state is None:
        best_model_state = deepcopy(model.state_dict())
        best_ddi_rate = ddi_values[-1] if len(ddi_values) > 0 else 0

    best_model_path = os.path.join(
        save_dir,
        'Epoch_{}_JA_{:.4f}_DDI_{:.4f}.model'.format(best_epoch, best_ja, best_ddi_rate)
    )

    torch.save(best_model_state, open(best_model_path, 'wb'))
    return best_model_path, best_epoch, best_ja, best_ddi_rate


if __name__ == '__main__':
    set_seed()
    args = parse_args()
    logging.info(args)

    # set logger
    if args.test:
        args.note = 'test of ' + args.log_dir_prefix
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log' + str(log_save_id) + '_' + args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("PID: %s", os.getpid())
    logging.info(args)

    if not torch.cuda.is_available() or args.cuda < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.cuda}')

    dataset = args.dataset
    data_path = f'../data/output/{dataset}/records_final.pkl'
    voc_path = f'../data/output/{dataset}/voc_final.pkl'
    ddi_adj_path = f'../data/output/{dataset}/ddi_A_final.pkl'
    ddi_mask_path = f'../data/output/{dataset}/ddi_mask_H.pkl'  
    molecule_path = f'../data/output/{dataset}/atc3toSMILES.pkl'
    substruct_smile_path = f'../data/output/{dataset}/substructure_smiles.pkl'

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
        data_train = [[visit] for patient in data_train for visit in patient]
        data_eval = [[visit] for patient in data_eval for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]

    average_projection, smiles_list = \
        buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)

    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': args.mol_num_layer,
        'emb_dim': args.mol_emb_dim,
        'graph_pooling': args.mol_graph_pooling,
        'drop_ratio': args.mol_drop_ratio,
        'gnn_type': args.mol_gnn_type,
        'virtual_node': args.mol_virtual_node
    }

    if args.embedding:
        substruct_para, substruct_forward = None, None
    else:
        with open(substruct_smile_path, 'rb') as Fin:
            substruct_smiles_list = dill.load(Fin)

        substruct_graphs = graph_batch_from_smile(substruct_smiles_list)
        substruct_forward = {'batched_data': substruct_graphs.to(device)}
        substruct_para = {
            'num_layer': args.sub_num_layer,
            'emb_dim': args.sub_emb_dim,
            'graph_pooling': args.sub_graph_pooling,
            'drop_ratio': args.sub_drop_ratio,
            'gnn_type': args.sub_gnn_type,
            'virtual_node': args.sub_virtual_node
        }

    model = DifMTTModel(
        global_para=molecule_para, substruct_para=substruct_para,
        emb_dim=args.dim, global_dim=args.dim, substruct_dim=args.dim,
        substruct_num=ddi_mask_H.shape[1], voc_size=voc_size,
        use_embedding=args.embedding, device=device, dropout=args.dp,
        num_intents=args.num_intents if args.multi_intent else 1,
        use_multi_intent=args.multi_intent
            ,
        use_diffusion=args.diffusion,
        diffusion_steps=args.diff_steps,
        diffusion_beta_start=args.diff_beta_start,
        diffusion_beta_end=args.diff_beta_end,
        diffusion_ddi_weight=args.diff_ddi_weight
    ).to(device)
    sbcl_state = SBCLState(
        enabled=bool(args.sbcl),
        diag_voc_size=voc_size[0],
        device=device,
        args=args
    )
    intent_state = IntentState(
        enabled=bool(args.multi_intent),
        num_intents=args.num_intents,
        intent_dim=args.dim,
        device=device,
        args=args
    )

    drug_data = {
        'substruct_data': substruct_forward,
        'mol_data': molecule_forward,
        'ddi_mask_H': ddi_mask_H,
        'tensor_ddi_adj': ddi_adj,
        'average_projection': average_projection
    }

    _run_t0 = time.time()

    if args.test:
        rec_results_path = save_dir + '/' + 'rec_results'
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        Test(model, dataset, model_path, device, data_test, voc_size, drug_data, rec_results_path, args=args)
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        optimizer = Adam(model.parameters(), lr=args.lr)
        best_model_path, best_epoch, best_ja, best_ddi_rate = Train(
            model, dataset, device, data_train, data_eval, voc_size, drug_data,
            optimizer, save_dir, args.coef, args.target_ddi, EPOCH=args.epochs,
            sbcl_state=sbcl_state, intent_state=intent_state, args=args
        )
        if args.test_after_train:
            rec_results_path = save_dir + '/' + 'rec_results'
            Test(model, dataset, best_model_path, device, data_test, voc_size, drug_data, rec_results_path, args=args)

    _run_t1 = time.time()
    print(_format_hms(_run_t1 - _run_t0))
