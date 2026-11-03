#!/usr/bin/env python3
"""Keep a reference copy, then remove only generated stage outputs."""
from pathlib import Path
import argparse,shutil
ROOT=Path(__file__).resolve().parent
NAMES={'endpoints':'article_missing_scale_degfp_endpoints_20260902',
 'uncertainty':'five_axis_uncertainty_20261027','neighbors':'adjacent_local_optima_20261027'}

def main():
 p=argparse.ArgumentParser();p.add_argument('stages',nargs='+',choices=list(NAMES));a=p.parse_args()
 for key in a.stages:
  path=ROOT/'run_outputs'/NAMES[key]
  reference=ROOT/'reference_results/original_run_outputs'/NAMES[key]
  if not path.exists():continue
  if path.is_symlink():raise RuntimeError('Refusing to clear a symlink: '+str(path))
  if not reference.exists():
   reference.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(reference))
  else:shutil.rmtree(path)
  print('Prepared fresh output directory:',path)

if __name__=='__main__':main()
