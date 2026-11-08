"""Normalize remaining ungrouped output examples in operational documentation."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = {'VALIDATION.md', 'CALIBRATION_VALIDATION.md', 'SELFMOTION_VALIDATION.md',
              'PYTHON_SPIKING_VALIDATION.md', 'SPIKING_MT_VALIDATION.md', 'M1_VALIDATION.md',
              'PYTHON_V1_VALIDATION.md', 'MT_STAGE_VALIDATION.md', 'FILE_ORGANIZATION.md'}
INPUTS = {
    'v1_gratings_01': 'grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01',
    'v1_textures_01': 'texture/texture_calibration__80x80__f000040__dt50ms__8dirs__v01',
    'selfmotion_v1_01': 'selfmotion/selfmotion_suite__80x80__f000040__dt50ms__v01',
    'matlab_bw_grating_01': 'grating/bw_grating_suite__80x80__f000080__dt50ms__v01',
    'validation_sequences_02': 'sequence_preparation/manual__sequence_generation__r02',
}


def replace(match):
    name = match.group(1)
    if name in ('recordings', 'visualizations', 'comparisons'):
        return match.group(0)
    if name in INPUTS:
        return 'input/stimuli/' + INPUTS[name]
    if name.startswith('cpp_v1_build') or name.startswith('build_'):
        return 'build/' + name
    if name == 'export_fixture':
        return 'build/' + name
    if any(token in name for token in ('check', 'suite', '8dirs')):
        return 'experiments/validation/manual__' + name + '__r01'
    return 'experiments/scratch/manual__' + name + '__r01'


def main():
    files = [ROOT / 'README.md', ROOT / 'matlab/README.md', ROOT / 'srcipts/README.md', ROOT / 'matlab/generate_bw_grating_suite.m']
    files += [p for p in (ROOT / 'docs').glob('*.md') if p.name not in HISTORICAL]
    for p in files:
        text = p.read_text(encoding='utf-8')
        updated = re.sub(r'(?<![\w/])outputs/([a-zA-Z0-9_]+)', replace, text)
        updated = updated.replace('`srcipts/bw_grating_output/sequence_时间戳/`', '`input/stimuli/grating/grating_sequence__宽x高__f帧数__dt间隔ms__v01__t时间戳/`')
        updated = re.sub(r'experiments/scratch/(manual__[a-zA-Z0-9_]*(?:check|suite|8dirs)[a-zA-Z0-9_]*__r01)', r'experiments/validation/\1', updated)
        if updated != text:
            p.write_text(updated, encoding='utf-8')


if __name__ == '__main__':
    main()
