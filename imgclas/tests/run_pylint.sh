#!/usr/bin/env bash

# Script to run pylint static code analysis
#
# Created on Mon Jul  2 16:16:37 2018
# @author: vkozlov

# search for python files 4 levels down. Apply message template for output. Pass all parameters.
pylint $(find . -maxdepth 4 -name "*.py") --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" "$@"
