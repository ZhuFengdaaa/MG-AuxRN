name=$1
polyaxon run -f polyaxonfile.yml --description=$name \
    -P name=$name
