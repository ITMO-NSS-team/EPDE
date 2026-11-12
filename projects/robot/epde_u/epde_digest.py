"""Digest of the EPDE package a run imports (Nov 5 2026, weak-enhance plan): the package may be edited while our
runs go on, so every run records {HEAD, `git status --porcelain -- epde`, blake2b over the sorted epde/**/*.py bytes}
at its start and end. Two runs compare only under an equal digest. Read-only: this module writes nothing.
python epde_digest.py  -> prints the digest as JSON."""
import hashlib
import json
import os
import subprocess

# Nov 12: under EPDE_REPO (a frozen `git archive` snapshot, no .git) the digest hashes the snapshot's epde/ and reads
# its HEAD from SNAPSHOT_HEAD (written at export); porcelain is the constant 'snapshot'.
REPO = os.environ.get('EPDE_REPO') or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def _git(*args):
    try:
        return subprocess.run(['git', '-C', REPO] + list(args), capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001 -- the digest must never stop a run
        return 'git-error: %s' % exc


def source_hash(root: str = os.path.join(REPO, 'epde')) -> str:
    h = hashlib.blake2b(digest_size=16)
    paths = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        paths.extend(os.path.join(base, f) for f in files if f.endswith('.py'))
    for path in sorted(paths):
        h.update(os.path.relpath(path, root).replace('\\', '/').encode())
        with open(path, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()


def digest() -> dict:
    snap = os.path.join(REPO, 'SNAPSHOT_HEAD')
    if os.path.exists(snap):
        with open(snap, encoding='utf-8') as fh:
            return dict(head=fh.read().strip(), porcelain='snapshot', source=source_hash())
    porcelain = _git('status', '--porcelain', '--', 'epde')
    return dict(head=_git('rev-parse', 'HEAD'),
                porcelain=hashlib.blake2b(porcelain.encode(), digest_size=8).hexdigest(),
                source=source_hash())


if __name__ == '__main__':
    print(json.dumps(digest()))
