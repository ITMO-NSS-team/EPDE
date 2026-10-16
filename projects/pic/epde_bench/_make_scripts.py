"""Write scripts/<script>.py, one entry point per data folder, from the registry
(run after changing a loader's truth or notes: python -m epde_bench._make_scripts).
The original research scripts in data/<folder>/ are not touched."""
import textwrap
from pathlib import Path

from .datasets import REGISTRY, load
from .paths import PIC_DIR

#: folders that had no script of their own before
NEW_FOLDERS = {'duffing', 'jhtdb', 'daisy', 'sst', 'dp'}

FOLDERS = {
    'ode': ('ode.py', ['ode'], 'Forced oscillator with time-dependent damping'),
    'vdp': ('vdp.py', ['vdp'], 'Van der Pol oscillator'),
    'duffing': ('duffing.py', ['duffing'], 'Forced Duffing oscillator'),
    'lv': ('lv.py', ['lv'], 'Lotka-Volterra predator-prey system'),
    'lorenz': ('lorenz.py', ['lorenz'], 'Lorenz-63 system'),
    'ac': ('ac.py', ['ac'], 'Allen-Cahn equation'),
    'burgers': ('burgers.py', ['burgers', 'burgers_inviscid'], 'Burgers equation (viscous and inviscid)'),
    'wave': ('wave.py', ['wave'], 'Wave equation'),
    'kdv': ('kdv.py', ['kdv', 'kdv_cossin', 'kdv_homogen', 'kdv_sga'], 'Korteweg-de Vries equation, four records'),
    'ks': ('ks.py', ['ks'], 'Kuramoto-Sivashinsky equation'),
    'pde_compound': ('pde_compound.py', ['pde_compound'], 'Nonlinear diffusion u_t = (u u_x)_x'),
    'pde_divide': ('pde_divide.py', ['pde_divide'], 'PDE with a 1/x coefficient'),
    'ns': ('ns.py', ['ns'], 'Navier-Stokes, flow past a cylinder'),
    'heat_solar': ('heat_solar.py', ['heat_solar_1d', 'heat_solar_2d'], 'Heat in soil under solar forcing'),
    'heat_laser': ('heat_laser.py', ['heat_laser'], 'Heat equation with a moving laser source'),
    'jhtdb': ('jhtdb.py', ['jhtdb_plane'], 'Isotropic turbulence slice (Johns Hopkins Turbulence Database)'),
    'darcy': ('darcy.py', ['darcy'], 'Darcy flow'),
    'daisy': ('daisy.py', ['robot_arm', 'ballbeam'], 'DaISy system-identification records'),
    'sst': ('sst.py', ['sst'], 'Sea surface temperature (ESA CCI L4)'),
    'dp': ('dp.py', ['dp_encoder', 'pend_single', 'dp_sim', 'dp_video'], 'Pendulums: real rig and simulation'),
}


def _wrap(text, indent):
    return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)


def describe(name):
    spec = REGISTRY[name]
    lines = [f'  {name}: {spec.title}']
    try:
        problem = load(name)
        if problem.truth:
            lines.append('      truth:')
            lines += [f'          {eq}' for eq in problem.truth]
            if problem.truth_alternatives:
                lines.append(f'          (+ {len(problem.truth_alternatives)} equivalent form(s) accepted)')
        else:
            lines.append('      truth:  unknown (not scored)')
        lines.append(_wrap('notes:  ' + problem.notes, '      '))
        lines.append(f'      shape:  {problem.shape}, axes {", ".join(problem.axis_names)}')
    except FileNotFoundError:
        lines.append('      DATA FILES MISSING -- the loader is ready for when they are added.')
    lines.append(f'      data:   projects/pic/data/{spec.files}')
    lines.append(f'      config: projects/pic/configs/{name}.yaml (on top of configs/_base.yaml)')
    lines.append(f'      suite:  {spec.suite}')
    return '\n'.join(lines)


def script_text(folder, script, names, title):
    path = f'projects/pic/scripts/{script}'
    original = f'projects/pic/data/{folder}/{script}'
    multi = len(names) > 1
    pick = f' --dataset {names[1]}' if multi else ''
    body = '\n\n'.join(describe(n) for n in names)
    return f'''"""{title}.

Data set{'s' if multi else ''} in this folder (all of them: projects/pic/DATASETS.md):

{body}

Usage -- the same on Windows, Linux and macOS, from any directory:

    python {path}{pick}                    # one noise-free run, seed 0
    python {path}{pick} --noise 5 --seed 3
    python {path}{pick} --variant poly     # method variants: configs/variants.yaml
    python {path}{pick} --check            # does the known law hold on the data?
    python {path}{pick} --plot             # look at the data
    python {path} --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('{names[0]}')
    cfg = load_config('{names[0]}')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder{'' if folder in NEW_FOLDERS else f', {original},'} is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main({names!r}))
'''


if __name__ == '__main__':
    for folder, (script, names, title) in FOLDERS.items():
        target = PIC_DIR / 'scripts' / script
        target.write_text(script_text(folder, script, names, title), encoding='utf-8', newline='\n')
        print('wrote', target)
