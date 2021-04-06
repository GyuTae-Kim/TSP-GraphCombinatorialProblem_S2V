# Graph Combinatorial Problem

### Requirement
- python 3(>=3.5)
- tensorflow(>=2.0)
- numpy
- pandas

## Tutorial

### Clone
- Clone this repository:
```
git clone https://github.com/GyuTae-Kim/TSP-GraphCombinatorialProblem_S2V.git
cd TSP-GraphCombinatorialProblem_S2V
```

### Train Model (+ Test)
```
python main.py --config configs.yaml
```

### Test Model (Only)
```
python main.py --config configs.yaml --test_only
```

### (Optionally) Test Model with Generated Dataset

- Generate Dataset
```
python tool/test_data_gen.py --config configs.yaml --save_path data/test_data --one_per_one
```

- Test Model
```
python main.py --test_only --config configs.yaml --test_data_path data/test_data
```

## Reference
```
https://papers.nips.cc/paper/2017/file/d9896106ca98d3d05b8cbdf4fd8b13a1-Paper.pdf
```
