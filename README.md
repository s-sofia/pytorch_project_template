## How To Use
To run inference (evaluate the model or save predictions):
   ```bash
git clone https://github.com/s-sofia/pytorch_project_template.git
cd pytorch_project_template
git checkout example/asr
pip install -r requirements.txt
python3 inference.py HYDRA_CONFIG_ARGUMENTS
   ```

## How To Receive The Model
The actual model architecture is in the file baseline_model.py, branch example/asr. It takes 10 epochs to get the model.
   ```bash
python3 train.py
   ```

https://drive.google.com/file/d/1meQbb-FUj9sVAJtibQsRYqmjBNgrTVri/view?usp=sharing

## Report

https://wandb.ai/sekhova-sn-l/pytorch_template_asr_example/runs/lyqntbam

all runs
https://wandb.ai/sekhova-sn-l/pytorch_template_asr_example?nw=nwusersekhovasn

Done by Lisichkina Sofia
