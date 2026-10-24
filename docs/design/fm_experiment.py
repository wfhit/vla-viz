import sys,os,math,time,json,torch
sys.path.insert(0,"/home/frank/behavior_challenge_ws/behavior_challenge/src"); os.chdir("/home/frank/behavior_challenge_ws/behavior_challenge/src")
torch.set_num_threads(12)
from transformers import AutoProcessor
from data.behavior_action_rows import RowSet
from data.fused_batch import collate_fused
import model.fused_behavior_action as fba
from model.one_step_actions import create_sinusoidal_pos_embedding
import torch.nn.functional as F
from training.fused_dual_head_sft import SUPERSEDED_CKPT_KEYS, build_fused_model, fused_train_tensors
D="/home/frank/workspace/behavior_challenge_data"
m=build_fused_model(260,261,gradient_checkpointing=False)
ck=torch.load(f"{D}/outputs/k005b-task0-v3/checkpoints/step_130000/checkpoint.pt",map_location="cpu",weights_only=False)
m.load_state_dict({k:v for k,v in ck["model"].items() if k not in SUPERSEDED_CKPT_KEYS},strict=True); del ck
m=m.float().eval()
rs=RowSet("/home/frank/behavior_challenge_ws/outputs/train-data-task0-donehold-v3",f"{D}/dataset/behavior-1k-knack",vocab_cache_path=f"{D}/outputs/k005b-task0-v3-vocab-cache.json")
proc=AutoProcessor.from_pretrained("/home/frank/workspace/model/qwen/Qwen3-VL-2B-Instruct"); proc.tokenizer.padding_side="right"
b=collate_fused([rs[0]],proc)
tok=getattr(proc,"image_token_id",None) or proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
with torch.no_grad(): T=fused_train_tensors(m,b,tok,"cpu")
tg=b["action_target"][0].float()
TV={"t":1.0}
orig=m._suffix_embeds
def patched(s75,x1):
    bb,k,_=x1.shape; pw=m.projections.action_time_mlp_out.in_features
    te=create_sinusoidal_pos_embedding(torch.full((bb,),TV["t"]),pw,min_period=4e-3,max_period=4.0)
    se=m.projections.state_proj(s75); ae=m.projections.action_in_proj(x1)
    at=m.projections.action_time_mlp_out(F.silu(m.projections.action_time_mlp_in(torch.cat([ae,te[:,None].expand(bb,k,pw)],-1))))
    return torch.cat([se[:,None],at],1),te
m._suffix_embeds=patched
def vel(x,t):
    TV["t"]=t; d=dict(T); d["x1"]=x[None]
    with torch.no_grad(): return m(**d).action_velocity[0].float()
res={}
x1=T["x1"][0].float()
t0=time.time(); v=vel(x1,1.0); print("fwd",time.time()-t0,flush=True)
c=x1-v; res["seed_base"]={"mse":((c-tg)**2).mean().item()}
chunks=[c]
for s in [1,2,3]:
    n=torch.randn(16,23,generator=torch.Generator().manual_seed(1000+s))
    cc=n-vel(n,1.0); chunks.append(cc)
C=torch.stack(chunks)
res["across_noise_chunk_std"]=C.std(0).mean().item()
res["across_noise_x1_std"]=1.0
res["mean_chunk_mse"]=((C.mean(0)-tg)**2).mean().item()
res["per_seed_mse"]=[((cc-tg)**2).mean().item() for cc in chunks]
# counterfactual 10-step Euler t:1->0
N=10; x=x1.clone(); traj=[]
for k in range(N):
    t=1-k/N; vk=vel(x,t); x=x-(1/N)*vk
    traj.append({"t":t,"vnorm":vk.norm().item(),"mse_x_target":((x-tg)**2).mean().item()})
    print(traj[-1],flush=True)
res["euler10"]=traj
json.dump(res,open("/tmp/fmexp/res.json","w"),indent=1); print(json.dumps(res,indent=1))
