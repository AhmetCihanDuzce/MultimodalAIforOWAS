"""Exploratory nested model/view selection with a matched leg hierarchy ablation."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys, importlib.metadata
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from run_experiments import Benchmark, SEED
from modular_followup.models import CLASSES, HierarchicalLegRF, fit_node, make_candidates
from modular_followup.feature_interface import assemble, predict_fitted

OUT = ROOT / 'modular_followup'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_paths():
    return [OUT / n for n in ['protocol.json', 'models.py', 'feature_interface.py', 'run_modular.py']] + [ROOT / n for n in ['features/features.npz', 'leg_followup/geometry_features.npz', 'inputs/expert_consensus_input.csv', 'audit/paired_manifest.csv', 'run_experiments.py']]


def freeze():
    paths = {str(p.relative_to(ROOT)): sha(p) for p in checked_paths()}
    target = OUT / 'protocol_freeze.json'
    if target.exists():
        assert json.loads(target.read_text())['sha256'] == paths, 'Development inputs/code changed after freeze'
    else:
        assert not (OUT / 'results').exists(), 'Freeze must precede new result computation'
        target.write_text(json.dumps({'created_at_utc': datetime.now(timezone.utc).isoformat(), 'prior_pilot_and_leg_results_seen': True,
              'new_modular_accuracy_evaluated': False, 'new_worker_data_seen': False, 'sha256': paths}, indent=2) + '\n')


class ModularStudy(Benchmark):
    def __init__(self):
        super().__init__()
        self.arrays = {k: self.z[k] for k in self.z.files}
        geometry = np.load(ROOT / 'leg_followup/geometry_features.npz', allow_pickle=False)
        self.arrays.update({k: geometry[k] for k in geometry.files})
        self.indices = np.arange(len(self.df))
        self.feature_cache = {}

    def features_for(self, candidate):
        key = (candidate['family'], candidate['view'])
        if key not in self.feature_cache:
            self.feature_cache[key] = assemble(self.arrays, candidate)
        return self.feature_cache[key]

    def fit(self, component, candidate, train):
        x, available = self.features_for(candidate)
        usable = train[available[train]]
        y = self.z['y_' + component]
        counts = {str(c): int((y[usable] == c).sum()) for c in CLASSES[component]}
        support = {'train_n': len(train), 'usable_train_n': len(usable), 'class_counts': counts}
        if not len(usable):
            support['status'] = 'no_usable_training_input'
            return None, support
        if candidate['hierarchy']:
            model = HierarchicalLegRF(**candidate['parameters']).fit(x[usable], y[usable])
            support['hierarchy_training_support'] = model.training_support_
        else:
            model = fit_node(x[usable], y[usable], candidate['family'], candidate['parameters'])
        support['status'] = 'single_class_training_fallback' if sum(n > 0 for n in counts.values()) == 1 else 'fitted'
        return model, support

    def infer(self, model, candidate, indices):
        x, available = self.features_for(candidate)
        return predict_fitted(model, x[indices], available[indices], candidate['hierarchy'])

    def context(self, name, train, test, block, fold, refit=False):
        directory = OUT / 'results' / name
        directory.mkdir(parents=True, exist_ok=True)
        cache = directory / ('context_' + str(fold) + '.joblib')
        if cache.exists():
            saved = joblib.load(cache)
            assert saved['train'] == train.tolist() and saved['test'] == test.tolist()
            assert saved['name'] == name and saved['fold'] == fold and saved['block_seconds'] == block
            assert saved['protocol_freeze_sha256'] == sha(OUT / 'protocol_freeze.json')
            return saved
        inner = self.splits(train, block, 3, SEED + fold)
        outputs, selections, candidate_logs, oofs, fitted_models = {}, {}, {}, {}, {}
        for component, classes in CLASSES.items():
            candidates = make_candidates(component) + (make_candidates(component, hierarchy=True) if component == 'legs' else [])
            logs = []
            prediction_map = {}
            for candidate in candidates:
                predictions = np.zeros(len(self.df), dtype=np.int16)
                scores, support_logs = [], []
                for inner_fold, (it, iv, ip) in enumerate(inner, 1):
                    model, support = self.fit(component, candidate, it)
                    predictions[iv], _ = self.infer(model, candidate, iv)
                    scores.append(float(f1_score(self.z['y_' + component][iv], predictions[iv], labels=classes, average='macro', zero_division=0)))
                    support_logs.append({'inner_fold': inner_fold, **support})
                logs.append({**candidate, 'fold_macro_f1': scores, 'mean_macro_f1': float(np.mean(scores)), 'training_support': support_logs})
                prediction_map[candidate['id']] = predictions
            choose = lambda pool: max(pool, key=lambda item: item['mean_macro_f1'])
            flat = [v for v in logs if not v['hierarchy']]
            mapping = {'modular_flat': choose(flat)}
            if component == 'legs':
                mapping['modular_RF_flat'] = choose([v for v in flat if v['family'] == 'combined_leg_RF'])
                mapping['modular_RF_hierarchy'] = choose([v for v in logs if v['hierarchy']])
            else:
                mapping['modular_RF_flat'] = mapping['modular_flat']
                mapping['modular_RF_hierarchy'] = mapping['modular_flat']
            for family in ['pose_RF', 'pose_RBF_SVM', 'embedding_linear_SVM']:
                for view in ['side', 'front', 'paired']:
                    mapping['homogeneous_' + family + '_' + view] = choose([v for v in flat if v['family'] == family and v['view'] == view])
            chosen_models = {}
            for system, candidate in mapping.items():
                ident = candidate['id']
                if ident not in chosen_models:
                    model, support = self.fit(component, candidate, train)
                    pred, coarse = self.infer(model, candidate, test)
                    chosen_models[ident] = (pred, coarse, support)
                    if refit:
                        fitted_models[component + ':' + ident] = {'model': model, 'candidate': candidate, 'training_support': support}
                pred, coarse, support = chosen_models[ident]
                outputs.setdefault(system, {})['pred_' + component] = pred
                if component == 'legs':
                    outputs[system]['pred_leg_family'] = coarse
                selections.setdefault(system, {})[component] = {'candidate_id': ident, 'candidate': candidate, 'full_training_support': support}
            candidate_logs[component] = logs
            oofs[component] = prediction_map
            print('COMPLETED', name, fold, component, 'candidates', len(candidates), flush=True)
        saved = {'name': name, 'fold': fold, 'block_seconds': block, 'protocol_freeze_sha256': sha(OUT / 'protocol_freeze.json'), 'train': train.tolist(), 'test': test.tolist(),
           'inner': [{'train': a.tolist(), 'validation': b.tolist(), 'purged': c.tolist()} for a,b,c in inner],
           'predictions': outputs, 'selections': selections, 'candidate_logs': candidate_logs, 'inner_oof_predictions': oofs}
        joblib.dump(saved, cache, compress=3)
        if refit:
            frozen = OUT / 'frozen_workflow'
            frozen.mkdir(exist_ok=True)
            joblib.dump(fitted_models, frozen / 'models.joblib', compress=3)
        return saved

    def run(self, mode):
        if mode in ['60', '30']:
            block = int(mode)
            name = 'blocks_' + mode + 's'
            contexts = [(fold, tr, te, removed) for fold,(tr,te,removed) in enumerate(self.splits(self.indices, block, 5, SEED), 1)]
        elif mode == 'transfer':
            block = 60
            name = 'recording_transfer'
            contexts = [(fold, np.flatnonzero(self.ds == source), np.flatnonzero(self.ds == target), np.array([], int)) for fold,(source,target) in enumerate([('coverall','everyday'), ('everyday','coverall')], 101)]
        elif mode == 'freeze':
            self.freeze_external()
            return
        else:
            raise ValueError(mode)
        records, split_log = [], []
        for fold, train, test, removed in contexts:
            saved = self.context(name, train, test, block, fold)
            split_log.append({'fold': fold, 'train': train.tolist(), 'test': test.tolist(), 'purged': removed.tolist(), 'inner': saved['inner']})
            for system, predictions in saved['predictions'].items():
                q = self.df.iloc[test].copy()
                q['observation_index'] = test
                q['fold'] = fold
                q['method'] = system
                q['view'] = 'inner_selected' if system.startswith('modular_') else system.split('_')[-1]
                if mode == 'transfer':
                    q['source_recording'] = str(self.ds[train[0]])
                for component in CLASSES:
                    q['selected_' + component] = saved['selections'][system][component]['candidate_id']
                for col, values in predictions.items():
                    q[col] = values
                records.append(q)
        directory = OUT / 'results' / name
        result = pd.concat(records, ignore_index=True)
        result.to_csv(directory / 'predictions.csv', index=False)
        (directory / 'splits.json').write_text(json.dumps({'block_seconds': block, 'purge_seconds': 10, 'folds': split_log}, indent=2) + '\n')
        self.report(result, directory)
        self.hierarchy_report(result, directory)
        print('RUN_COMPLETE', name, flush=True)

    def hierarchy_report(self, predictions, directory):
        q = predictions[predictions.method == 'modular_RF_hierarchy'].copy()
        family = lambda x: 0 if x in [2,3] else 1 if x in [4,5] else 2 if x == 7 else -1
        q['true_leg_family'] = q.target_legs.map(family)
        q['gate_correct'] = q.pred_leg_family == q.true_leg_family
        q['leg_correct'] = q.pred_legs == q.target_legs
        q.to_csv(directory / 'hierarchy_error_rows.csv', index=False)
        rows = []
        for c in CLASSES['legs']:
            r = q[q.target_legs == c]
            rows.append({'class': c, 'support': len(r), 'unresolved': int((r.pred_legs == 0).sum()),
               'wrong_family': int(((r.pred_leg_family >= 0) & ~r.gate_correct).sum()),
               'correct_family_wrong_subcode': int((r.gate_correct & ~r.leg_correct).sum()), 'correct_leg': int(r.leg_correct.sum())})
        pd.DataFrame(rows).to_csv(directory / 'hierarchy_errors.csv', index=False)

    def freeze_external(self):
        frozen = OUT / 'frozen_workflow'
        frozen.mkdir(exist_ok=True)
        if (frozen / 'workflow.json').exists():
            return
        context_path = OUT / 'results/full_training/context_201.joblib'
        if context_path.exists() and not (frozen / 'models.joblib').exists():
            raise RuntimeError('Incomplete full-training checkpoint: review before rerunning')
        saved = self.context('full_training', self.indices, np.array([], dtype=int), 60, 201, refit=True)
        systems = {name: {comp: comp + ':' + info['candidate_id'] for comp,info in select.items()} for name,select in saved['selections'].items()}
        config = {'version': 'modular_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
           'primary_system': 'modular_flat', 'systems': systems, 'classes': CLASSES, 'load': 1,
           'training_worker_ids': ['W001'], 'external_validation': 'not run; no new-worker data',
           'selection_context': 'all existing observations used only for future frozen inference; inner threefold selection; no full-training performance claim',
           'models_file': 'models.joblib', 'models_sha256': sha(frozen / 'models.joblib'),
           'software_versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'mediapipe', 'Pillow']},
           'training_input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'features/features.npz', ROOT/'leg_followup/geometry_features.npz', ROOT/'audit/paired_manifest.csv', ROOT/'audit/image_manifest.csv']},
           'development_freeze_sha256': sha(OUT / 'protocol_freeze.json'),
           'primary_endpoint': 'full-code exact agreement on all eligible paired-images-only consensus observations, unresolved retained',
           'source_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [OUT/'models.py', OUT/'feature_interface.py', ROOT/'extract_features.py', ROOT/'leg_followup/leg_features.py']},
           'model_assets': json.loads((ROOT/'models/model_assets.json').read_text())}
        (frozen / 'workflow.json').write_text(json.dumps(config, indent=2) + '\n')
        print('FROZEN_EXTERNAL_WORKFLOW_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['60', '30', 'transfer', 'freeze', 'all'], default='all')
    args = parser.parse_args()
    freeze()
    study = ModularStudy()
    with threadpool_limits(limits=1):
        for mode in (['60', '30', 'transfer', 'freeze'] if args.mode == 'all' else [args.mode]):
            study.run(mode)


if __name__ == '__main__':
    main()
