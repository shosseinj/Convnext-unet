"""Evaluate a probability ensemble of completed ablation seed checkpoints."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from ablation_artifacts import DATASETS, atomic_write_json
from ablation_registry import get_experiment
from checkpoint_management import load_checkpoint_file, strip_thop_state
from evaluation_core import make_loader, evaluate_loader, count_parameters, measure_complexity
from one_seed_models import build_experiment_model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment_name',required=True); p.add_argument('--seed_root',type=Path,required=True); p.add_argument('--seeds',nargs='+',type=int,default=[42,6543,7777]); p.add_argument('--data_path',type=Path,default=Path('data')); p.add_argument('--encoder_weights',type=Path,default=Path('convnext_tiny_22k_1k_384.pth')); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch_size',type=int,default=8); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); args=p.parse_args()
    config=get_experiment(args.experiment_name); device=torch.device(args.device); models=[]
    for seed in args.seeds:
        sd=args.seed_root/f'seed_{seed}'; ck=load_checkpoint_file(sd/'best_checkpoint.pth')
        if ck.get('training_complete') is not True or ck.get('architecture') != config.to_dict(): raise RuntimeError(f'Invalid checkpoint seed {seed}')
        m=build_experiment_model(config,args.encoder_weights,device); m.load_state_dict(strip_thop_state(ck['model_state_dict']),strict=True); m.eval(); models.append(m)
    results={}
    for d in DATASETS:
        loader,count=make_loader(args.data_path,d,args.batch_size,0)
        # Average logits across independently trained seeds; reuse metric implementation via a wrapper.
        class Ensemble(torch.nn.Module):
            def forward(self,x): return torch.stack([m(x) for m in models]).mean(0)
        metrics=evaluate_loader(Ensemble(),loader,device,threshold=0.45); metrics['samples']=count; results[d]=metrics
        print(f'[{d}] mDice={metrics["mDice"]:.4f} mIoU={metrics["mIoU"]:.4f}',flush=True)
    trainable,total=count_parameters(models[0]); payload={'experiment_name':config.name,'seeds':args.seeds,'threshold':0.45,'tta':False,'ensemble':'mean_logits','trainable_parameters':trainable,'total_parameters':total,**measure_complexity(models[0]),'results':results}
    atomic_write_json(args.output,payload); print(f'Ensemble summary saved: {args.output}')
if __name__=='__main__': main()
