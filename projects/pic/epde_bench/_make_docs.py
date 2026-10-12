"""Write projects/pic/DATASETS.md from the registry: python -m epde_bench._make_docs

Titles and notes come from the loaders in datasets.py, so the document and
``Problem.summary()`` never disagree.
"""

from .datasets import REGISTRY, load
from .paths import PIC_DIR
from .problem import KINDS

SOURCES = {'synthetic': 'synthetic', 'real': 'measured'}
SUITES = {
    'core': 'synthetic with a known law; the main benchmark tables',
    'extended': 'synthetic with a known law, slower or less standard',
    'real': 'measurements',
    'other': 'loads, but is not part of the benchmark',
}


def main():
    rows, sections = [], []
    for suite in SUITES:
        for name, spec in REGISTRY.items():
            if spec.suite != suite:
                continue
            title = spec.title
            try:
                p = load(name)
                shape = ' x '.join(map(str, p.shape))
                variables = ', '.join(p.variables)
                truth = 'known' if p.truth else 'unknown'
                if p.truth and p.truth_alternatives:
                    truth += f' (+{len(p.truth_alternatives)} alt.)'
                body = [f'- **class:** {KINDS[p.kind]}; **source:** {SOURCES[p.source]}; **suite:** {suite}',
                        f'- **axes:** {", ".join(p.axis_names)}; **shape:** {shape}; **variables:** {variables}']
                if p.truth:
                    body.append('- **law:**')
                    body += [f'  - `{eq}`' for eq in p.truth]
                    for alt in p.truth_alternatives:
                        body.append('  - also accepted: ' + '; '.join(f'`{eq}`' for eq in alt))
                if p.token_groups or p.extra_arrays:
                    labels = [lbl for _, t, _ in p.token_groups for lbl in t] + list(p.extra_arrays)
                    body.append(f'- **extra tokens:** {", ".join(labels)}')
                if p.derivs is not None:
                    body.append('- **derivatives:** supplied with the data and passed to EPDE as given')
                body.append(f'- **notes:** {p.notes}')
            except FileNotFoundError:
                shape, variables, truth = '-', '-', 'no files'
                body = ['- **status:** the data files are missing; loading and search are not verified.']
            rows.append(f'| `{name}` | {suite} | {KINDS[spec.kind]} | {SOURCES[spec.source]} | '
                        f'{shape} | {variables} | {truth} | {title} |')
            sections.append(f'### `{name}` -- {title}\n\n' + '\n'.join(body))
    text = ['# Data sets in projects/pic/data', '',
            'Generated from the data-set loaders; edit the loaders, not this file. The name in the '
            'first column is what the command line, the scripts and the notebooks use to refer to a '
            'record.', '',
            'Suites: ' + '; '.join(f'**{k}** -- {v}' for k, v in SUITES.items()) + '.', '',
            'Equations are written in EPDE text form: `dx0` is time, `dx1`, `dx2`, ... are the space '
            'axes in the order given under "axes"; `x{power: p, dim: k}` is the k-th coordinate to the '
            'power p. Only the set of terms is scored; coefficients are for reference.', '',
            '| name | suite | class | source | shape | variables | law | title |',
            '|---|---|---|---|---|---|---|---|', *rows, '',
            '## Details', '', '\n\n'.join(sections), '']
    (PIC_DIR / 'DATASETS.md').write_text('\n'.join(text), encoding='utf-8', newline='\n')
    print('wrote', PIC_DIR / 'DATASETS.md')


if __name__ == '__main__':
    main()
