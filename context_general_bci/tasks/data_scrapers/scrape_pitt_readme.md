# A guide to scraping Pitt (and possibly Chicago) data

These datasets are the bulk of the human data. A variety of scripts come in for varied purposes. We provide a survey. Note that these are matlab scripts as python wrappers for reading archival data format (Quicklogger) are not yet available.

These scripts are intended to be run on some Pitt workstation with fast access to the storage server; the processing is done on the workstation, and we then transfer all to compute server.
- `pull_broad.m` - Major scrape (runtime ~few days) for all historical.
    - We add major paradigms of interest by spot-checking historical logs for major paradigms (not ideal, JY can't remember how this list was created).
    - We remove blatantly useless data with stopwords from test log comments.
    - We exclude sessions with stimulation data.
    - Implementation is quite inefficient, we do it query by query.
    - `poll_broad.m` - provide a specific starting date for `pull_broad.m`. Used to keep pretraining dataset ~up to date across participants. That is, our current continual pretraining strategy is non-existent, we just pool all data all the time.
Last poll: 11/1! TODO CRON-ify...?
- `pull_bmi01.m` - One-off for pulling pre-standardized pathing data from BMI01.
- `pull_manual.m` - Convenience wrapper for allowing manual selection of data files for processing. Used for pulling spot-comparisons of recent data.
- `prepThinDNNPayload.m` - core logic for extracting DNN payload from all archived data.
To pull a specific set that we know the IDs for, use `loadData`, i.e.
```matlab
data = loadData([138], [10], 'subject', 'P3Home', 'class', 'MotorExperiments');
prepThinDNNPayload(data, 'D:/data/manual/');
```