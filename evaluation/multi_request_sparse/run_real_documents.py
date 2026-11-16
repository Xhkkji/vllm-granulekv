"""Run one real scenario/cache state through the existing benchmark launcher."""
import argparse
from pathlib import Path
import subprocess
from real_documents import load_scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--cache', choices=('cold', 'hot'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    _, scenario, prefix, suffixes = load_scenario(args.manifest, args.scenario)
    subprocess.run(['bash', str(Path(__file__).with_name('run.sh')),
                    str(len(prefix)), str(max(map(len, suffixes))), 'same',
                    str(args.output_dir.resolve()), '--real-manifest', str(args.manifest.resolve()),
                    '--scenario', args.scenario, '--formal-output-tokens', '512',
                    '--repeats', '1', '--gate-cache', args.cache, '--cache-states', args.cache,
                    '--service-timeout', '1800', '--no-progress-timeout', '120', '--performance'], check=True)


if __name__ == '__main__':
    main()
