## Install Dependencies

This part can be run either locally or on an AWS EC2 instance.
Please use Python3.7 and install the dependencies in the `requirements.txt`
file. We recommend creating a virtual environment with conda, e.g.

```bash
conda create -n cs224r-meta-rl python=3.7
# conda init bash
cd hw4/meta_rl/
conda activate cs224r-meta-rl
pip install -r requirements.txt
export PYTHONPATH=/home/paulemile/Documents/cs224r-deep-reinforcement-learning-hw4/hw4/meta_rl
```

## Run the Code

python scripts/rl2.py rl2_cpu_v2 -b environment=\"map\" --force_overwrite
python scripts/dream.py dream_cpu -b environment=\"map\" --force_overwrite

To run the DREAM code, invoke the following command:

```
python3 scripts/dream.py exp_name -b environment=\"map\" --force_overwrite
```

This will create a directory `experiments/exp_name`, which will contain:

- A tensorboard subdirectory at `experiments/exp_name/tensorboard`, which logs
  statistics, such as accumulated returns vs. number of training episodes, and
  also vs. number of training steps.
- A visualization subdirectory at `experiments/exp_name/visualize`, which will
  contain videos of the learned agent.
- A checkpoints subdirectory at `experiments/exp_name/checkpoints`, which will
  periodically save model checkpoints.
- Metadata about the run, such as the configs used.

You can pass different values for `exp_name` as convenient.

To run the RL^2 code, similarly run:

```
python3 rl2.py exp_name -b environment=\"map\"
```
