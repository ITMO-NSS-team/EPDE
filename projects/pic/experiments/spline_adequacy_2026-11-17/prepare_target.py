from adequacy_context import *

source=(ROOT/'selection_bootstrap.py').read_text(encoding='utf-8')
start=source.index('BOOTJOBS=');end=source.index('\n',start)
source=source[:start]+"BOOTJOBS=[dict(cohort='bootstrap',name='ode',profile='positive',seed=s) for s in range(3200,3456)]"+source[end:]
source=source.replace("'selection_bootstrap.py','bootstrap_protocol.txt'", "'selection_bootstrap.py','target_bootstrap.py','bootstrap_protocol.txt','target_protocol.txt'")
source=source.replace("'bootstrap_", "'target_")
# The replacement above also changes protocol reference; restore both protocols.
source=source.replace("'target_bootstrap.py','target_protocol.txt','target_protocol.txt'", "'target_bootstrap.py','bootstrap_protocol.txt','target_protocol.txt'")
old="active=[key for key in keys if read(ROOT/'target_anchors'/(key+'.json'))['model'] is not None];jobs="
new="active=sorted([key for key in keys if read(ROOT/'target_anchors'/(key+'.json'))['info']['selection']['selected_p']==24],key=lambda key:read(ROOT/'target_anchors'/(key+'.json'))['job']['seed'])[:8];dump(ROOT/'target_selection.json',dict(status='FROZEN',selected=active,scanned=keys,source_sha256=bsrc(),input_sha256={str(ROOT/'target_anchors'/(key+'.json')):sha(ROOT/'target_anchors'/(key+'.json')) for key in keys}));jobs="
assert old in source;source=source.replace(old,new)
source=source.replace('pool.submit(freeze,key) for key in keys','pool.submit(freeze,key) for key in active').replace('pool.submit(evaluate,key) for key in keys','pool.submit(evaluate,key) for key in active')
source=source.replace("for key in keys}))\n        for f in as_completed([pool.submit(evaluate", "for key in active}))\n        for f in as_completed([pool.submit(evaluate")
(ROOT/'target_bootstrap.py').write_text(source,encoding='utf-8')
audit=(ROOT/'audit_bootstrap.py').read_text(encoding='utf-8').replace('from selection_bootstrap import *','from target_bootstrap import *').replace("'bootstrap_","'target_")
audit=audit.replace('keys=[bkey(j) for j in BOOTJOBS]',"keys=read(ROOT/'target_selection.json')['selected']")
audit=audit.replace("'audit_bootstrap.py','audit_adequacy.py','selection_bootstrap.py'", "'audit_target.py','audit_adequacy.py','target_bootstrap.py'")
audit=audit.replace("'of48'","'of'+str(len(keys))")
(ROOT/'audit_target.py').write_text(audit,encoding='utf-8')
print('Prepared targeted conditional cohort; original follow-up unchanged.')
