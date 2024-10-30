# Install the public NDT3 evaluation datasets from NDT3 to paths matching those expected in context_registry.
# Assumes `./data` is initialized (symlinked if desired) and installing libs (e.g. dandi) are installed via `data_requirements.txt`
# Note this doesn't create the particular splits used (see split_eval.py, still manual currently...)

cd data

# FALCON
mkdir falcon
cd falcon
dandi download https://dandiarchive.org/dandiset/000954/draft
dandi download https://dandiarchive.org/dandiset/000950/draft
dandi download https://dandiarchive.org/dandiset/000941/draft
dandi download https://dandiarchive.org/dandiset/000953/draft
cd ..

# RTT
mkdir odoherty_rtt
cd odoherty_rtt
zenodo_get 3854034
cd ..

# NLB (not actually used right now)
mkdir nlb
cd nlb
dandi download DANDI:000128/0.220113.0400
dandi download DANDI:000138/0.220113.0407
dandi download DANDI:000139/0.220113.0408
dandi download DANDI:000140/0.220113.0408
cd ..


echo "The following dryad files should be downloaded manually:"
echo: "deo bimanual: https://datadryad.org/stash/dataset/doi:10.5061/dryad.sn02v6xbb"
echo "mender_fingerctx: https://datadryad.org/stash/dataset/doi:10.5061/dryad.p2ngf1vtn"
echo "miller tasks (e.g. Center-Out): https://datadryad.org/stash/dataset/doi:10.5061/dryad.cvdncjt7n"