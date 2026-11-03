"""Verify the pinned upstream source and the narrow allowed customization scope."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit():
    info = json.loads((ROOT / 'UPSTREAM.json').read_text())
    baseline = info['official_file_sha256']
    changed = []
    for name, expected in baseline.items():
        path = ROOT / name
        if not path.is_file():
            raise ValueError('Missing official file: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    if sorted(changed) != info['modified_upstream_files']:
        raise ValueError('Unexpected changes to official source: ' + repr(changed))
    if any(name.startswith('stage1/') for name in changed):
        raise ValueError('Stage 1 must remain identical to upstream')
    readme = (ROOT / 'README.md').read_text()
    marker = info.get('readme_original_retained_after_marker')
    if marker:
        if readme.count(marker) != 1 or hashlib.sha256(readme.split(marker, 1)[1].encode()).hexdigest() != baseline['README.md']:
            raise ValueError('Original README must remain intact below the project introduction')
    extras = [str(p.relative_to(ROOT)) for p in (ROOT / 'stage1').rglob('*')
              if p.is_file() and p.suffix in ('.py', '.json', '.yaml') and str(p.relative_to(ROOT)) not in baseline]
    extensions = info.get('extensions', {})
    allowed_stage1 = {name for name in extensions.get('stage1_class_fine', {}).get('modules', [])
                      if name.startswith('stage1/')}
    if set(extras) != allowed_stage1:
        raise ValueError('Unexpected Stage-1 modules: ' + repr(extras))
    stage2_extras = {str(p.relative_to(ROOT)) for p in (ROOT / 'stage2').rglob('*.py')
                     if str(p.relative_to(ROOT)) not in baseline}
    if stage2_extras != {'stage2/mario_llm_stage2/compat.py', 'stage2/mario_llm_stage2/subset.py'}:
        raise ValueError('Unexpected Stage-2 modules: ' + repr(stage2_extras))
    stage2_before = json.loads((ROOT / 'STAGE2_BEFORE_EXTENSIONS.json').read_text())
    for name, expected in stage2_before.items():
        content = (ROOT / name).read_bytes()
        if name == 'stage2/mario_llm_stage2/cli.py':
            # The user subsequently changed only the epoch cap and patience in
            # both Movies variants. Preserve the original extension snapshot.
            content = content.replace(b'"--epochs", type=int, default=8',
                                      b'"--epochs", type=int, default=5', 1)
            content = content.replace(b'"--patience", type=int, default=2',
                                      b'"--patience", type=int, default=5', 1)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Stage 2 changed while adding Stage-1/interstage extensions: ' + name)
    for name in ('stage2/mario_llm_stage2/neighbor_selector.py',
                 'stage2/mario_llm_stage2/neighbors.py',
                 'scripts/llama_class_refine.py', 'scripts/train_neighbor_selector.py'):
        if (ROOT / name).exists():
            raise ValueError('Experimental module present: ' + name)
    # Undo only the two documented runtime fixes, then verify the ENTIRE original
    # model source. This covers architecture, prompts, losses and Top-10 ranking.
    text = (ROOT / 'stage2/mario_llm_stage2/model.py').read_text()
    text = text.replace('from .compat import answer_logits_kwargs\n', '', 1)
    text = text.replace('**answer_logits_kwargs(LlamaForCausalLM.forward, logits_to_keep),',
                        'logits_to_keep=logits_to_keep,', 1)
    text = text.replace('''        outputs = self.llm.generate(
            inputs_embeds=inputs.unsqueeze(0),''', '''        position_ids = torch.arange(
            0, inputs.size(0), dtype=torch.long, device=self.device
        ).unsqueeze(0)
        outputs = self.llm.generate(
            inputs_embeds=inputs.unsqueeze(0),''', 1)
    text = text.replace('''            attention_mask=attention,
            max_new_tokens=self.max_new_tokens,''', '''            attention_mask=attention,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,''', 1)
    if hashlib.sha256(text.encode()).hexdigest() != baseline['stage2/mario_llm_stage2/model.py']:
        raise ValueError('Model differs from upstream beyond the two documented compatibility fixes')
    return dict(upstream=info['commit'], original_stage1_files=sum(n.startswith('stage1/') for n in baseline),
                modified_official_files=changed, experimental_modules_present=bool(extensions),
                authorized_extensions=[key for key, value in extensions.items() if isinstance(value, dict)],
                stage2_unchanged_by_extensions=True,
                later_stage2_settings_changes=info.get('stage2_run_setting_changes_after_extensions', {}),
                dataset=info['dataset'], stage2_epochs=info['stage2_epochs'],
                original_model_recovered_after_reversing_compatibility=True)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
