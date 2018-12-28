# Backends

### Snorkel Databases
This provides a generic wrapper class for using Snorkel v0.7 databases in MeTal.  We make several assumptions when using this class:
- The snorkel database is fully initialized (e.g., candidates, splits, probabilistic and gold labels are pre-defined).
- The application is a standard unary/binary relation extraction task.
- The dataset fits in local memory. 

Usage example:
```
db_conn_str = "cdr.db"
candidate_def = ['ChemicalDisease', ['chemical', 'disease']]
train, dev, test = SnorkelDataset.splits(db_conn_str, candidate_def)
```