"""Print a compact saved calibration report without running the neural model."""
import argparse
import json
from pathlib import Path

from research.daily_coarse_graining.daily_operator.feedback_loss_geometry import calibration_readout

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--canopy-weight', type=float)
    parser.add_argument('--litterpart-weight', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (args.canopy_weight is None) != (args.litterpart_weight is None):
        parser.error('supply both weights together, or neither')
    weights = None if args.canopy_weight is None else {
        'base': 1., 'canopy_stock_change': args.canopy_weight, 'litterpart_feedback': args.litterpart_weight}
    result = calibration_readout(json.loads(args.report.read_text()), coefficients=weights)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)
