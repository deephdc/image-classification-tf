import os
import textwrap
import builtins

import yaml

homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("file",os.path.abspath(__file__))
print("dir",os.path.dirname(os.path.abspath(__file__)))
print("home",homedir)
conf_path = os.path.join(homedir, 'etc', 'config.yaml')
with open(conf_path, 'r') as f:
    CONF = yaml.safe_load(f)
