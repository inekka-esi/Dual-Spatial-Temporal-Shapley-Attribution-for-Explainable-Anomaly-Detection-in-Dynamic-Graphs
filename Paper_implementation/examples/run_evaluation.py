"""
Dual Spatial-Temporal Shapley Attribution
for Explainable Anomaly Detection in Dynamic Graphs
=====================================================
Authors: Iyad Assaad Nekka, Hamida Seba,
         Khaled-Walid Hidouci, Karima Amrouche
ESI Algiers / University Claude Bernard Lyon 1

This script runs the full evaluation pipeline and produces:
  TABLE 2 — Detection AUC-ROC
  TABLE 3 — Explainability introduced (N/A vs Ours)
  TABLE 4 — Fidelity across edge categories
  TABLE 5 — Sensitivity to top-k budget

Usage (Google Colab, T4 GPU):
    Upload the xtaddy/ folder to /content/xtaddy/, then:
    !python examples/run_evaluation.py

Runtime: ~3 hours on T4 (training + full expanded evaluation).
"""

import os, sys, subprocess, shutil, json
import torch, numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics

# ── 1. Environment setup ──────────────────────────────────────────────────────
if not os.path.exists('/content/TADDY_pytorch'):
    subprocess.run(['git', 'clone',
                    'https://github.com/yuetan031/TADDY_pytorch.git',
                    '/content/TADDY_pytorch'])
os.chdir('/content/TADDY_pytorch')
sys.path.insert(0, '/content/TADDY_pytorch')
sys.path.insert(0, '/content')

os.system('pip install transformers networkx scipy scikit-learn numpy matplotlib -q')

for d in ['data/interim', 'data/percent', 'data/eigen',
          'data/processed/uci', 'data/processed/btc_alpha',
          'data/processed/btc_otc']:
    os.makedirs(d, exist_ok=True)
for ds in ['uci', 'btc_alpha', 'btc_otc']:
    p = f'data/interim/{ds}'
    if os.path.isdir(p): shutil.rmtree(p)

print("Environment ready.")

# ── 2. Patch TADDY for Python 3.12 compatibility ─────────────────────────────
with open('codes/Component.py', 'w') as f:
    f.write('''
import torch, math
import torch.nn as nn
from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    def __init__(self, k=5, max_hop_dis_index=100, max_inti_pos_index=100,
                 hidden_size=32, num_hidden_layers=1, num_attention_heads=1,
                 intermediate_size=32, hidden_act="gelu", hidden_dropout_prob=0.5,
                 attention_probs_dropout_prob=0.3, initializer_range=0.02,
                 layer_norm_eps=1e-12, is_decoder=False, batch_size=256,
                 window_size=1, weight_decay=5e-4, **kwargs):
        super().__init__(**kwargs)
        self.max_hop_dis_index=max_hop_dis_index; self.max_inti_pos_index=max_inti_pos_index
        self.k=k; self.hidden_size=hidden_size; self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads; self.hidden_act=hidden_act
        self.intermediate_size=intermediate_size; self.hidden_dropout_prob=hidden_dropout_prob
        self.attention_probs_dropout_prob=attention_probs_dropout_prob
        self.initializer_range=initializer_range; self.layer_norm_eps=layer_norm_eps
        self.is_decoder=is_decoder; self.batch_size=batch_size
        self.window_size=window_size; self.weight_decay=weight_decay

class EdgeEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inti_pos_embeddings=nn.Embedding(config.max_inti_pos_index,config.hidden_size)
        self.hop_dis_embeddings =nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.time_dis_embeddings=nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.LayerNorm=nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
    def forward(self,init_pos_ids=None,hop_dis_ids=None,time_dis_ids=None):
        emb=(self.inti_pos_embeddings(init_pos_ids)+
             self.hop_dis_embeddings(hop_dis_ids)+
             self.time_dis_embeddings(time_dis_ids))
        return self.dropout(self.LayerNorm(emb))

class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_heads=config.num_attention_heads
        self.head_dim=config.hidden_size//config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.query=nn.Linear(config.hidden_size,config.hidden_size)
        self.key  =nn.Linear(config.hidden_size,config.hidden_size)
        self.value=nn.Linear(config.hidden_size,config.hidden_size)
        self.out  =nn.Linear(config.hidden_size,config.hidden_size)
        self.dropout=nn.Dropout(config.attention_probs_dropout_prob)
        self.LayerNorm=nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
    def forward(self,hidden_states,attention_mask=None,head_mask=None,**kwargs):
        B,T,D=hidden_states.shape; H,HD=self.num_heads,self.head_dim
        Q=self.query(hidden_states).view(B,T,H,HD).transpose(1,2)
        K=self.key(hidden_states).view(B,T,H,HD).transpose(1,2)
        V=self.value(hidden_states).view(B,T,H,HD).transpose(1,2)
        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(HD)
        attn=self.dropout(torch.softmax(scores,dim=-1))
        ctx=torch.matmul(attn,V).transpose(1,2).contiguous().view(B,T,D)
        return self.LayerNorm(self.out(ctx)+hidden_states), attn

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense1=nn.Linear(config.hidden_size,config.intermediate_size)
        self.dense2=nn.Linear(config.intermediate_size,config.hidden_size)
        self.act=nn.GELU()
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm=nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
    def forward(self,x):
        return self.LayerNorm(self.dense2(self.dropout(self.act(self.dense1(x))))+x)

class TransformerLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention=SelfAttention(config)
        self.feedforward=FeedForward(config)
    def forward(self,hidden_states,attention_mask=None,head_mask=None,**kwargs):
        attn_out,attn_w=self.attention(hidden_states,attention_mask,head_mask)
        return self.feedforward(attn_out), attn_w

class TransformerEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layer=nn.ModuleList([TransformerLayer(config)
                                  for _ in range(config.num_hidden_layers)])
    def forward(self,hidden_states,attention_mask=None,head_mask=None,**kwargs):
        if head_mask is None: head_mask=[None]*len(self.layer)
        all_attn=[]
        for i,layer in enumerate(self.layer):
            hidden_states,attn_w=layer(hidden_states,attention_mask,head_mask[i])
            all_attn.append(attn_w)
        return hidden_states, all_attn
''')

with open('codes/BaseModel.py', 'w') as f:
    f.write('''
import torch.nn as nn
from codes.Component import EdgeEncoding, TransformerEncoder

class BaseModel(nn.Module):
    data = None
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.embeddings=EdgeEncoding(config)
        self.encoder=TransformerEncoder(config)
        self.pooler=nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size),nn.Tanh())
    def forward(self,init_pos_ids,hop_dis_ids,time_dis_ids,head_mask=None):
        if head_mask is None: head_mask=[None]*self.config.num_hidden_layers
        emb=self.embeddings(init_pos_ids=init_pos_ids,
                            hop_dis_ids=hop_dis_ids,
                            time_dis_ids=time_dis_ids)
        enc,all_attn=self.encoder(emb,head_mask=head_mask)
        return enc, self.pooler(enc[:,0,:]), all_attn
    def forward_from_embeddings(self, emb):
        enc, all_attn = self.encoder(emb)
        return enc, all_attn
    def run(self): pass
''')

with open('codes/DynADModel.py', 'w') as f:
    f.write('''
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, time, numpy as np
from sklearn import metrics
from codes.BaseModel import BaseModel
from codes.utils import dicts_to_embeddings, compute_batch_hop, compute_zero_WL

class DynADModel(nn.Module):
    """
    TADDY: Transformer-based Anomaly Detector for DYnamic graphs.
    Liu et al., TKDE 2022.

    Sigmoid clarification:
    cls_y returns RAW LOGITS. sigmoid() is applied exactly once outside
    this class when computing anomaly scores.
    """
    lr=0.001; weight_decay=5e-4; max_epoch=500

    def __init__(self, config, args):
        super().__init__()
        self.args=args; self.config=config
        self.transformer=BaseModel(config)
        self.cls_y=nn.Linear(config.hidden_size,1)  # returns logits
        self.weight_decay=config.weight_decay

    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, idx=None):
        """Returns (raw_logits, attn_weights). Caller applies sigmoid()."""
        enc,pooled,all_attn=self.transformer(init_pos_ids,hop_dis_ids,time_dis_ids)
        seq=sum(enc[:,i,:] for i in range(self.config.k+1))/float(self.config.k+1)
        return self.cls_y(seq), all_attn

    def forward_from_embeddings(self, emb):
        """
        Forward pass from pre-computed dense embeddings.
        Used by the explainability framework after coalition masking.
        Returns (raw_logits, attn_weights).
        """
        enc, all_attn = self.transformer.forward_from_embeddings(emb)
        seq = sum(enc[:,i,:] for i in range(self.config.k+1))/float(self.config.k+1)
        return self.cls_y(seq), all_attn

    def generate_embedding(self,edges):
        num_snap=len(edges)
        WL_dict=compute_zero_WL(self.data["idx"],np.vstack(edges[:7]))
        batch_hop_dicts=compute_batch_hop(
            self.data["idx"],edges,num_snap,
            self.data["S"],self.config.k,self.config.window_size)
        return dicts_to_embeddings(self.data["X"],batch_hop_dicts,WL_dict,num_snap)

    def negative_sampling(self,edges):
        node_list=self.data["idx"]; num_node=node_list.shape[0]; neg=[]
        for snap_edge in edges:
            ne=snap_edge.copy(); n=ne.shape[0]
            fi=node_list[np.random.choice(num_node,n)]
            fp=np.random.choice(2,n).tolist()
            ne[np.arange(n),fp]=fi; neg.append(ne)
        return neg

    def evaluate(self,trues,preds):
        aucs={i:metrics.roc_auc_score(trues[i],preds[i])
              for i in range(len(self.data["snap_test"]))}
        return aucs,metrics.roc_auc_score(np.hstack(trues),np.hstack(preds))

    def train_model(self,max_epoch):
        opt=optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        _,_,hop_emb,int_emb,time_emb=self.generate_embedding(self.data["edges"])
        _,wl_emb,_,_,_=self.generate_embedding(self.data["edges"])
        for epoch in range(max_epoch):
            t0=time.time()
            neg=self.negative_sampling(
                self.data["edges"][:max(self.data["snap_train"])+1])
            _,_,hop_neg,int_neg,time_neg=self.generate_embedding(neg)
            self.train(); loss_total=0; count=0
            for snap in self.data["snap_train"]:
                if wl_emb[snap] is None: continue
                ii=torch.vstack((int_emb[snap],int_neg[snap]))
                hh=torch.vstack((hop_emb[snap],hop_neg[snap]))
                tt=torch.vstack((time_emb[snap],time_neg[snap]))
                y=torch.hstack((self.data["y"][snap].float(),
                                torch.ones(int_neg[snap].size(0))))
                opt.zero_grad()
                out,_=self.forward(ii,hh,tt)
                loss=F.binary_cross_entropy_with_logits(out.squeeze(),y)
                loss.backward(); opt.step()
                loss_total+=loss.detach().item(); count+=1
            print(f"  Epoch {epoch+1:03d} | loss {loss_total/max(count,1):.4f} "
                  f"| {time.time()-t0:.1f}s")
            if (epoch+1)%self.args.print_feq==0:
                self.eval(); preds=[]
                for snap in self.data["snap_test"]:
                    with torch.no_grad():
                        p,_=self.forward(int_emb[snap],hop_emb[snap],time_emb[snap])
                        preds.append(torch.sigmoid(p).squeeze().numpy())
                y_test=[self.data["y"][s].numpy() for s in self.data["snap_test"]]
                _,auc=self.evaluate(y_test,preds)
                print(f"  >>> AUC: {auc:.4f}")

    def run(self): self.train_model(self.max_epoch)
''')

for mod in list(sys.modules):
    if 'codes' in mod: del sys.modules[mod]

from codes.Component import MyConfig
from codes.DynADModel import DynADModel
from codes.DynamicDatasetLoader import DynamicDatasetLoader
from codes.Settings import Settings

print("All patches applied and imports verified.")

# ── 3. Prepare datasets ───────────────────────────────────────────────────────
for dataset in ['uci', 'btc_alpha', 'btc_otc']:
    if os.path.exists(f'data/percent/{dataset}_0.5_0.1.pkl'):
        print(f"  {dataset}: already prepared."); continue
    r = subprocess.run(
        ['python', '0_prepare_data.py', '--dataset', dataset,
         '--anomaly_per', '0.1', '--train_per', '0.5'],
        capture_output=True, text=True)
    print(f"  {dataset}: {'OK' if r.returncode==0 else 'FAILED — '+r.stderr[-100:]}")

# ── 4. Import framework ───────────────────────────────────────────────────────
from xtaddy import XTADDYExplainer, summarise_explanations, print_tables

# ── 5. Main evaluation loop ───────────────────────────────────────────────────
SESSION = {}

for dataset in ['uci', 'btc_alpha', 'btc_otc']:
    print(f"\n{'='*62}")
    print(f"  DATASET: {dataset.upper()}")
    print(f"{'='*62}")

    np.random.seed(1); torch.manual_seed(1)

    data_obj = DynamicDatasetLoader()
    data_obj.dataset_name=dataset; data_obj.k=5; data_obj.window_size=2
    data_obj.anomaly_per=0.1; data_obj.train_per=0.5
    data_obj.load_all_tag=False; data_obj.compute_s=True

    config = MyConfig(k=5, window_size=2, hidden_size=32, intermediate_size=32,
                      num_attention_heads=2, num_hidden_layers=2, weight_decay=5e-4)

    class Args:
        anomaly_per=0.1; train_per=0.5; neighbor_num=5; window_size=2
        embedding_dim=32; num_hidden_layers=2; num_attention_heads=2
        lr=0.001; weight_decay=5e-4; seed=1; print_feq=10
    Args.dataset=dataset; Args.max_epoch=100; args=Args()

    model = DynADModel(config, args)
    model.lr=0.001; model.max_epoch=100
    setting_obj = Settings(); setting_obj.prepare(data_obj, model)
    loaded_data = data_obj.load(); model.data=loaded_data

    print(f"  Snaps:{loaded_data['num_snap']} | "
          f"Train:{loaded_data['snap_train']} | Test:{loaded_data['snap_test']}")

    print(f"\n  Training TADDY — 100 epochs")
    model.train_model(100)

    _,_,hop_emb,int_emb,time_emb = model.generate_embedding(loaded_data['edges'])
    model.eval(); preds=[]; y_test=[]
    for snap in loaded_data['snap_test']:
        with torch.no_grad():
            p,_ = model.forward(int_emb[snap], hop_emb[snap], time_emb[snap])
            preds.append(torch.sigmoid(p).squeeze().numpy())
        y_test.append(loaded_data['y'][snap].numpy())
    det_auc = metrics.roc_auc_score(np.hstack(y_test), np.hstack(preds))
    print(f"\n  Detection AUC: {det_auc:.4f}  (post-hoc, unchanged)")

    SESSION[dataset] = dict(
        model=model, loaded_data=loaded_data,
        int_emb=int_emb, hop_emb=hop_emb, time_emb=time_emb,
        preds=preds, y_test=y_test, det_auc=det_auc,
    )

    # Run framework on top-50 TP
    explainer = XTADDYExplainer(model, num_samples=150, top_k=5)
    top50 = explainer.explain_top_edges(
        loaded_data, int_emb, hop_emb, time_emb, preds, y_test, n=50)
    SESSION[dataset]['top50_results'] = top50
    SESSION[dataset]['top50_summary'] = summarise_explanations(top50)

# ── 6. Print tables ───────────────────────────────────────────────────────────
print_tables(SESSION)

# ── 7. Save results ───────────────────────────────────────────────────────────
output = {}
for ds in SESSION:
    s = SESSION[ds]['top50_summary']
    output[ds] = {
        'detection_auc':     SESSION[ds]['det_auc'],
        'fidelity':          s['fidelity_mean'],
        'sufficiency':       s['sufficiency_mean'],
        'comprehensiveness': s['comprehensiveness_mean'],
        'sparsity':          s['sparsity_mean'],
    }

with open('xtaddy_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nAll evaluations complete.")
print("Results saved to xtaddy_results.json")
