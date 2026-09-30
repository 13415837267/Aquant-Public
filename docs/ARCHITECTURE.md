# Architecture

    AKShare / Eastmoney
            |
            v
    update_candidates.py
      | universe rules
      | factor ranking
      | top-30 selection
            |
            v
    data/candidates.json
            |
       +----+----+
       |         |
       v         v
    Next.js   JSON API
       |
       v
    Published web UI

GitHub Actions schedule: 10:00 UTC = 18:00 Beijing time.

The private repository is the canonical research surface. Only a reviewed, release-safe runtime snapshot is maintained in the public repository.

The runtime deliberately stops at candidate generation. Broker/OMS execution is a later boundary and should be implemented as a separate service.
