"""Run baseline, spatial-only, competition-only and combined MT controls."""
from run_mt_spatial_controls import VARIANT_SWITCHES, parse_args, run


if __name__ == '__main__':
    run(parse_args(competition=True), variants=tuple(VARIANT_SWITCHES))
