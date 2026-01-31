from .gnn import GNNGraph

import math
import torch
import torch.nn as nn
import torch.nn.functional as F





class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.size(0), 1), device=device)], dim=-1)
        return emb


class DiffusionDenoiser(nn.Module):

    def __init__(self, x_dim: int, cond_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or (x_dim * 4)
        self.time_emb = SinusoidalTimeEmbedding(x_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim + x_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        inp = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(inp)


class GaussianDiffusion(nn.Module):

    def __init__(self, steps: int = 50, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.steps = int(steps)
        betas = torch.linspace(beta_start, beta_end, self.steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)                       # (T,)
        self.register_buffer("alphas", alphas)                     # (T,)
        self.register_buffer("alphas_cumprod", alphas_cumprod)     # (T,)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise
        """
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_om * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return (x_t - sqrt_om * eps) / (sqrt_ab + 1e-12)

    def predict_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return (x_t - sqrt_ab * x0) / (sqrt_om + 1e-12)

    def training_loss(
        self,
        denoiser: nn.Module,
        x0: torch.Tensor,
        cond: torch.Tensor,
        drug_emb: torch.Tensor = None,
        ddi_adj: torch.Tensor = None,
        ddi_weight: float = 0.0,
    ) -> torch.Tensor:

        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = denoiser(x_t, cond, t)
        loss = F.mse_loss(eps_pred, noise)

        if ddi_weight > 0 and (drug_emb is not None) and (ddi_adj is not None):
            # 用当前 x0_pred 产生药物概率，再计算 DDI risk
            x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)  # (B, D)
            logits = torch.matmul(x0_pred, drug_emb.t())          # (B, N)
            p = torch.sigmoid(logits)
            # pairwise interaction expectation: sum_{i,j} p_i p_j A_{ij}
            pair = torch.matmul(p.transpose(0, 1), p)             # (N, N) 
            ddi_pen = 0.0005 * pair.mul(ddi_adj).sum()
            loss = loss + float(ddi_weight) * ddi_pen

        return loss

    @torch.no_grad()
    def _ddi_guidance_x0(
        self,
        x0: torch.Tensor,
        drug_emb: torch.Tensor,
        ddi_adj: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:

        if guidance_scale <= 0:
            return x0
        logits = torch.matmul(x0, drug_emb.t())  # (B, N)
        p = torch.sigmoid(logits)                # (B, N)
        r = torch.matmul(p, ddi_adj)             # (B, N)
        w = torch.softmax(r, dim=-1)             # (B, N)
        g = torch.matmul(w, drug_emb)            # (B, D)
        x0_guided = x0 - guidance_scale * g
        return x0_guided

    @torch.no_grad()
    def p_sample(
        self,
        denoiser: nn.Module,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        t: int,
        drug_emb: torch.Tensor = None,
        ddi_adj: torch.Tensor = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:

        B = x_t.size(0)
        tt = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        eps = denoiser(x_t, cond, tt)
        x0 = self.predict_x0_from_eps(x_t, tt, eps)

        if (drug_emb is not None) and (ddi_adj is not None) and guidance_scale > 0:
            x0 = self._ddi_guidance_x0(x0, drug_emb, ddi_adj, guidance_scale)
            eps = self.predict_eps_from_x0(x_t, tt, x0)

        beta_t = self.betas[tt].unsqueeze(-1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[tt].unsqueeze(-1)
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alphas_cumprod[tt].unsqueeze(-1)

        model_mean = sqrt_recip_alpha_t * (x_t - beta_t * eps / (sqrt_one_minus_ab_t + 1e-12))

        if t == 0:
            return model_mean

        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return model_mean + sigma * noise

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        cond: torch.Tensor,
        x_dim: int,
        drug_emb: torch.Tensor = None,
        ddi_adj: torch.Tensor = None,
        guidance_scale: float = 0.0,
        sample_steps: int = None,
    ) -> torch.Tensor:
        steps = int(sample_steps) if sample_steps is not None else self.steps
        steps = min(steps, self.steps)
        # 只做后 steps 个反向步（更快）
        start_t = self.steps - 1
        end_t = self.steps - steps

        x = torch.randn((cond.size(0), x_dim), device=cond.device)
        for t in range(start_t, end_t - 1, -1):
            x = self.p_sample(
                denoiser=denoiser,
                x_t=x,
                cond=cond,
                t=t,
                drug_emb=drug_emb,
                ddi_adj=ddi_adj,
                guidance_scale=guidance_scale,
            )
        return x

class MAB(torch.nn.Module):
    def __init__(
        self, Qdim, Kdim, Vdim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(MAB, self).__init__(*args, **kwargs)
        self.Vdim = Vdim
        self.number_heads = number_heads

        assert self.Vdim % self.number_heads == 0, \
            'the dim of features should be divisible by number_heads'

        self.Qdense = torch.nn.Linear(Qdim, self.Vdim)
        self.Kdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Vdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Odense = torch.nn.Linear(self.Vdim, self.Vdim)

        self.use_ln = use_ln
        if self.use_ln:
            self.ln1 = torch.nn.LayerNorm(self.Vdim)
            self.ln2 = torch.nn.LayerNorm(self.Vdim)

    def forward(self, X, Y):
        Q, K, V = self.Qdense(X), self.Kdense(Y), self.Vdense(Y)
        batch_size, dim_split = Q.shape[0], self.Vdim // self.number_heads

        Q_split = torch.cat(Q.split(dim_split, 2), 0)
        K_split = torch.cat(K.split(dim_split, 2), 0)
        V_split = torch.cat(V.split(dim_split, 2), 0)

        Attn = torch.matmul(Q_split, K_split.transpose(1, 2))
        Attn = torch.softmax(Attn / math.sqrt(dim_split), dim=-1)
        O = Q_split + torch.matmul(Attn, V_split)
        O = torch.cat(O.split(batch_size, 0), 2)

        O = O if not self.use_ln else self.ln1(O)
        O = self.Odense(O)
        O = O if not self.use_ln else self.ln2(O)

        return O


class SAB(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(SAB, self).__init__(*args, **kwargs)
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X):
        return self.net(X, X)


class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        fix_feat = torch.diag(fix_feat)
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)

        return O



class MultiIntentEncoder(nn.Module):

    def __init__(self, input_dim, intent_dim, num_intents, dropout=0.5):
        super(MultiIntentEncoder, self).__init__()
        self.num_intents = num_intents
        self.intent_dim = intent_dim
        self.intent_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, intent_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(intent_dim, intent_dim)
            ) for _ in range(num_intents)
        ])

        self.intent_weight_net = nn.Sequential(
            nn.Linear(input_dim, intent_dim),
            nn.ReLU(),
            nn.Linear(intent_dim, num_intents),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, patient_repr):

        squeeze_output = False
        if patient_repr.dim() == 1:
            patient_repr = patient_repr.unsqueeze(0)
            squeeze_output = True
        

        intent_reprs = []
        for k in range(self.num_intents):
            intent_k = self.intent_encoders[k](patient_repr)  # (batch_size, intent_dim)
            intent_reprs.append(intent_k)
        
        intent_reprs = torch.stack(intent_reprs, dim=1)  # (batch_size, num_intents, intent_dim)
        
        intent_weights = self.intent_weight_net(patient_repr)  # (batch_size, num_intents)
        
        if squeeze_output:
            intent_reprs = intent_reprs.squeeze(0)  # (num_intents, intent_dim)
            intent_weights = intent_weights.squeeze(0)  # (num_intents,)
        
        return intent_reprs, intent_weights


class IntentAwareSubstructRela(nn.Module):
    def __init__(self, intent_dim, substruct_num, num_intents):
        super(IntentAwareSubstructRela, self).__init__()
        self.num_intents = num_intents
        self.substruct_rela_per_intent = nn.ModuleList([
            nn.Linear(intent_dim, substruct_num)
            for _ in range(num_intents)
        ])
    
    def forward(self, intent_reprs):

        squeeze_output = False
        if intent_reprs.dim() == 2:
            intent_reprs = intent_reprs.unsqueeze(0)
            squeeze_output = True
        
        batch_size = intent_reprs.shape[0]
        substruct_weights = []
        
        for k in range(self.num_intents):
            intent_k = intent_reprs[:, k, :]  # (batch_size, intent_dim)
            weight_k = torch.sigmoid(self.substruct_rela_per_intent[k](intent_k))  # (batch_size, substruct_num)
            substruct_weights.append(weight_k)
        
        substruct_weights = torch.stack(substruct_weights, dim=1)  # (batch_size, num_intents, substruct_num)
        
        if squeeze_output:
            substruct_weights = substruct_weights.squeeze(0)  # (num_intents, substruct_num)
        
        return substruct_weights



class DifMTTModel(torch.nn.Module):
    def __init__(
        self, global_para, substruct_para, emb_dim, voc_size,
        substruct_num, global_dim, substruct_dim, use_embedding=False,
        device=torch.device('cpu'), dropout=0.5,
        num_intents=1, use_multi_intent=False,
        use_diffusion: bool = False,
        diffusion_steps: int = 50,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 0.02,
        diffusion_ddi_weight: float = 0.0,
        diffusion_denoiser_hidden: int = None,
        *args, **kwargs
    ):
        super(DifMTTModel, self).__init__(*args, **kwargs)
        self.device = device
        self.use_embedding = use_embedding
        self.num_intents = num_intents
        self.use_multi_intent = use_multi_intent and num_intents > 1
        self.emb_dim = emb_dim
        self.substruct_num = substruct_num


        self.use_diffusion = bool(use_diffusion)
        self.diffusion_ddi_weight = float(diffusion_ddi_weight)
        if self.use_diffusion:
            self.diffusion = GaussianDiffusion(
                steps=int(diffusion_steps),
                beta_start=float(diffusion_beta_start),
                beta_end=float(diffusion_beta_end),
            )
            self.diffusion_denoiser = DiffusionDenoiser(
                x_dim=emb_dim,
                cond_dim=emb_dim,
                hidden_dim=diffusion_denoiser_hidden,
                dropout=dropout if dropout > 0 else 0.0,
            )
        else:
            self.diffusion = None
            self.diffusion_denoiser = None

        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, emb_dim)
            )
        else:
            self.substruct_encoder = GNNGraph(**substruct_para)

        self.global_encoder = GNNGraph(**global_para)

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()

        self.sab = SAB(substruct_dim, substruct_dim, 2, use_ln=True)
        
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim)
        )
        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)

        if self.use_multi_intent:
            self.multi_intent_encoder = MultiIntentEncoder(
                input_dim=emb_dim * 4,
                intent_dim=emb_dim,
                num_intents=num_intents,
                dropout=dropout
            )
            self.intent_substruct_rela = IntentAwareSubstructRela(
                intent_dim=emb_dim,
                substruct_num=substruct_num,
                num_intents=num_intents
            )

            self.intent_aggregators = torch.nn.ModuleList([
                AdjAttenAgger(global_dim, substruct_dim, max(global_dim, substruct_dim))
                for _ in range(num_intents)
            ])

            self.intent_score_extractors = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(substruct_dim, substruct_dim // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(substruct_dim // 2, 1)
                ) for _ in range(num_intents)
            ])

        self.aggregator = AdjAttenAgger(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )
        score_extractor = [
            torch.nn.Linear(substruct_dim, substruct_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(substruct_dim // 2, 1)
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if self.use_embedding:
            torch.nn.init.xavier_uniform_(self.substruct_emb)

    def _encode_patient_base(self, patient_data):

        seq1, seq2 = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.rnn_dropout(self.embeddings[0](Idx1))
            repr2 = self.rnn_dropout(self.embeddings[1](Idx2))
            seq1.append(torch.sum(repr1, keepdim=True, dim=1))
            seq2.append(torch.sum(repr2, keepdim=True, dim=1))

        seq1 = torch.cat(seq1, dim=1)
        seq2 = torch.cat(seq2, dim=1)
        output1, hidden1 = self.seq_encoders[0](seq1)
        output2, hidden2 = self.seq_encoders[1](seq2)

        seq_repr = torch.cat([hidden1, hidden2], dim=-1)
        last_repr = torch.cat([output1[:, -1], output2[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
        
        return patient_repr  # (emb_dim * 4,)

    def _encode_patient(self, patient_data):
        """Encode patient visit sequence into query vector (unnormalized) and substructure weights."""
        patient_repr = self._encode_patient_base(patient_data)
        query = self.query(patient_repr)  # (emb_dim,)
        substruct_weight = torch.sigmoid(self.substruct_rela(query))  # (substruct_num,)
        return query, substruct_weight

    @torch.no_grad()
    def encode_patient(self, patient_data, normalize=True):
        """Get patient query embedding for contrastive learning (no grad)."""
        query, _ = self._encode_patient(patient_data)
        if normalize:
            query = F.normalize(query, dim=-1)
        return query

    @torch.no_grad()
    def encode_patient_intents(self, patient_data, normalize=True):

        if not self.use_multi_intent:
            query = self.encode_patient(patient_data, normalize=normalize)
            return query, query.unsqueeze(0), torch.ones(1, device=self.device)
        
        patient_repr = self._encode_patient_base(patient_data)
        intent_reprs, intent_weights = self.multi_intent_encoder(patient_repr)

        aggregated_intent = torch.sum(intent_weights.unsqueeze(-1) * intent_reprs, dim=0)  # (intent_dim,)
        
        if normalize:
            aggregated_intent = F.normalize(aggregated_intent, dim=-1)
            intent_reprs = F.normalize(intent_reprs, dim=-1)
        
        return aggregated_intent, intent_reprs, intent_weights

    def _get_substruct_embeddings(self, substruct_data):

        return self.sab(
            self.substruct_emb.unsqueeze(0) if self.use_embedding else
            self.substruct_encoder(**substruct_data).unsqueeze(0)
        ).squeeze(0)

    def _drug_set_embedding(self, y_true: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:

        y = y_true.float()
        denom = y.sum(dim=-1, keepdim=True).clamp_min(1.0)
        x0 = torch.matmul(y, drug_emb) / denom
        x0 = F.normalize(x0, dim=-1)
        return x0

    def forward(
        self, substruct_data, mol_data, patient_data,
        ddi_mask_H, tensor_ddi_adj, average_projection,
        # Diffusion (optional)
        y_true: torch.Tensor = None,
        return_diffusion_loss: bool = False,
        diffusion_infer: bool = False,
        diffusion_guidance_scale: float = 0.0,
        diffusion_fuse_alpha: float = 0.5,
        diffusion_sample_steps: int = None,
        return_query: bool = False, normalize_query: bool = True,
        return_intent_info: bool = False
    ):
        global_embeddings = self.global_encoder(**mol_data)                 # (num_drugs_raw, D)
        global_embeddings = torch.mm(average_projection, global_embeddings) # (num_drugs, D)
        substruct_embeddings = self._get_substruct_embeddings(substruct_data)  # (substruct_num, D)

        def _ddi_penalty_from_score(score_logits: torch.Tensor) -> torch.Tensor:
            neg_pred_prob = torch.sigmoid(score_logits)
            neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
            return 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()

        diffusion_loss = None

        if self.use_multi_intent:
            patient_repr = self._encode_patient_base(patient_data)
            intent_reprs, intent_weights = self.multi_intent_encoder(patient_repr)  # (K,D), (K,)
            intent_substruct_weights = self.intent_substruct_rela(intent_reprs)     # (K, substruct_num)

            intent_scores = []
            for k in range(self.num_intents):
                substruct_weight_k = intent_substruct_weights[k]
                molecule_embeddings_k = self.intent_aggregators[k](
                    global_embeddings, substruct_embeddings,
                    substruct_weight_k, mask=torch.logical_not(ddi_mask_H > 0)
                )
                score_k = self.intent_score_extractors[k](molecule_embeddings_k).t()  # (1, num_drugs)
                intent_scores.append(score_k)

            intent_scores = torch.stack(intent_scores, dim=0)  # (K,1,N)
            base_score = torch.sum(intent_weights.view(self.num_intents, 1, 1) * intent_scores, dim=0)  # (1,N)

            aggregated_intent = torch.sum(intent_weights.unsqueeze(-1) * intent_reprs, dim=0)  # (D,)
            cond_vec = F.normalize(aggregated_intent, dim=-1) if normalize_query else aggregated_intent

            score = base_score
            if diffusion_infer and self.use_diffusion:
                x0_hat = self.diffusion.sample(
                    denoiser=self.diffusion_denoiser,
                    cond=cond_vec.unsqueeze(0),
                    x_dim=self.emb_dim,
                    drug_emb=global_embeddings,
                    ddi_adj=tensor_ddi_adj,
                    guidance_scale=float(diffusion_guidance_scale),
                    sample_steps=diffusion_sample_steps,
                )  # (1,D)
                score_diff = torch.matmul(x0_hat, global_embeddings.t())  # (1,N)
                a = float(diffusion_fuse_alpha)
                score = (1.0 - a) * base_score + a * score_diff

            batch_neg = _ddi_penalty_from_score(score)

            if return_diffusion_loss and self.use_diffusion and (y_true is not None):
                x0 = self._drug_set_embedding(y_true, global_embeddings)  # (B,D)
                cond_b = cond_vec.unsqueeze(0).expand(x0.size(0), -1)
                diffusion_loss = self.diffusion.training_loss(
                    denoiser=self.diffusion_denoiser,
                    x0=x0,
                    cond=cond_b,
                    drug_emb=global_embeddings,
                    ddi_adj=tensor_ddi_adj,
                    ddi_weight=self.diffusion_ddi_weight,
                )
            elif return_diffusion_loss:
                diffusion_loss = torch.zeros((), device=self.device)

            # returns
            if return_intent_info:
                if normalize_query:
                    aggregated_intent = F.normalize(aggregated_intent, dim=-1)
                if return_diffusion_loss:
                    return score, batch_neg, diffusion_loss, aggregated_intent, intent_reprs, intent_weights
                return score, batch_neg, aggregated_intent, intent_reprs, intent_weights

            if return_query:
                if normalize_query:
                    aggregated_intent = F.normalize(aggregated_intent, dim=-1)
                if return_diffusion_loss:
                    return score, batch_neg, diffusion_loss, aggregated_intent
                return score, batch_neg, aggregated_intent

            if return_diffusion_loss:
                return score, batch_neg, diffusion_loss
            return score, batch_neg

        else:
            query, substruct_weight = self._encode_patient(patient_data)
            query_norm = F.normalize(query, dim=-1) if normalize_query else query

            molecule_embeddings = self.aggregator(
                global_embeddings, substruct_embeddings,
                substruct_weight, mask=torch.logical_not(ddi_mask_H > 0)
            )
            base_score = self.score_extractor(molecule_embeddings).t()  # (1,N)

            score = base_score
            if diffusion_infer and self.use_diffusion:
                x0_hat = self.diffusion.sample(
                    denoiser=self.diffusion_denoiser,
                    cond=query_norm.unsqueeze(0),
                    x_dim=self.emb_dim,
                    drug_emb=global_embeddings,
                    ddi_adj=tensor_ddi_adj,
                    guidance_scale=float(diffusion_guidance_scale),
                    sample_steps=diffusion_sample_steps,
                )
                score_diff = torch.matmul(x0_hat, global_embeddings.t())
                a = float(diffusion_fuse_alpha)
                score = (1.0 - a) * base_score + a * score_diff

            batch_neg = _ddi_penalty_from_score(score)

            if return_diffusion_loss and self.use_diffusion and (y_true is not None):
                x0 = self._drug_set_embedding(y_true, global_embeddings)
                cond_b = query_norm.unsqueeze(0).expand(x0.size(0), -1)
                diffusion_loss = self.diffusion.training_loss(
                    denoiser=self.diffusion_denoiser,
                    x0=x0,
                    cond=cond_b,
                    drug_emb=global_embeddings,
                    ddi_adj=tensor_ddi_adj,
                    ddi_weight=self.diffusion_ddi_weight,
                )
            elif return_diffusion_loss:
                diffusion_loss = torch.zeros((), device=self.device)

            if return_query:
                if return_diffusion_loss:
                    return score, batch_neg, diffusion_loss, query_norm
                return score, batch_neg, query_norm

            if return_diffusion_loss:
                return score, batch_neg, diffusion_loss
            return score, batch_neg
