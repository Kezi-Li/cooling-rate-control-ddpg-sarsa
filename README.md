# Reinforcement Learning for Cooling-Rate Control in Directed Energy Deposition

This repository contains the finite-difference cooling-rate environment and the
DDPG and SARSA training code used in the accompanying EAAI study.

## Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the DDPG training results

```bash
bash scripts/reproduce_ddpg.sh
```

This trains 10 independent policies for 150 episodes using seeds 0--9 and
writes the outputs to `Results/Results_DDPG`.

## Reproduce the SARSA training results

```bash
bash scripts/reproduce_sarsa.sh
```

This trains 10 independent policies for 800 episodes using seeds 0--9 and
writes the outputs to `Results/Results_SARSA`.

Both experiments are CPU-intensive and may require many hours. The number of
parallel workers can be changed in the two scripts to match the available CPU
and memory resources.
