# Project overview
----------------
This project finetunes SpeechT5 to the NBTale dataset of Norwegian speech.

## Dataset
The NBTale dataset can be downloaded from:
https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-31/

## TensorBoard
Launch TensorBoard on localhost to inspect loss curves from all saved runs:

```bash
python analyze_tensorboard.py models
python analyze_tensorboard.py models --host 127.0.0.1 --port 6006 --open-browser

# console summaries still available if needed
python analyze_tensorboard.py models --summary --tag "loss|learning_rate"

# background the server and return immediately
python analyze_tensorboard.py models --detach
```
