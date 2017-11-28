#!/bin/bash

set -eu

function ensure_venv {
  test -n "${VIRTUAL_ENV+x}" && return

  if [ ! -d venv ]; then
    virtualenv -p /usr/local/bin/python3.6 venv
    ./venv/bin/pip install -r requirements.txt
  fi

  if [ requirements.txt -nt venv ]; then
    ./venv/bin/pip install -r requirements.txt
    touch ./venv
  fi

  set +u
  source ./venv/bin/activate
  set -u
}

function ensure_custom_modules {
  export PYTHONPATH=`pwd`
}

function task_lint {
  PY_FILES=$(find . -name "*.py" -not -path "./venv/*")
  flake8 --ignore=E501 ${PY_FILES}
}

function task_format {
  ensure_venv
  PY_FILES=$(find . -name "*.py" -not -path "./venv/*")
  yapf -i ${PY_FILES}
}

function task_usage {
    echo "Usage: $0 preprocess-svcomp-data | format"
    exit 1
}

function task_preprocess_svcomp_data {
  ensure_venv
  ensure_custom_modules
  # task_lint
  python ui/preprocessing.py
}

CMD=${1:-}
shift || true
case ${CMD} in
  preprocess-svcomp-data) task_preprocess_svcomp_data ;;
  format) task_format ;;
  *) task_usage ;;
esac
