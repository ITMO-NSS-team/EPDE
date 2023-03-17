import pytest
import pycodestyle



style = pycodestyle.StyleGuide(quite=True)
result = style.check_files('C:\\Users\\User\\Desktop\\ntcv\\EPDE\\epde\\interface\\equation_translator.py')
assert result.total_errors==0, f"Found code style errors (and warnings)."