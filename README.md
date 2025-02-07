Mat2Seq
==========
This code accompanies the NeurIPS 2024 paper [Invariant Tokenization of Crystalline Materials for Language Model Enabled Generation](https://openreview.net/forum?id=18FGRNd0wZ). Forked from AIRS repo.
Mat2Seq is a framework for creating unique and complete crystal sequence representations, as well as constructing a material LLM capable of generating novel crystal structures with desired properties of interest.

Mat2Seq has been evaluated on the Perov-5, Carbon-24, MP-20 and MPTS-52 benchmarks. This document describes how to 
reproduce the benchmark evaluation experiments described in the paper. The Perov-5, Carbon-24 and MP-20 datasets are
from the [CDVAE repository](https://github.com/txie-93/cdvae/tree/f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea/data), 
while the MPTS-52 dataset is from the 
[DiffCSP repository](https://github.com/jiaor17/DiffCSP/tree/fd6f48cef306193c0fb678db785424abcdad6dfd/data).

To assess Mat2Seq on a benchmark, a sequence of steps is required to: 
1. pre-process the original benchmark dataset
2. tokenize the pre-processed CIF files
3. train the model
4. generate CIF files
5. post-process the generated CIF files
6. compute the benchmark metrics.

Below, we will provide instructions using the MP-20 benchmark as an example.

### 1. Pre-processing the original benchmark dataset

Prepare the benchmark CSV files:
```shell
python mat2seq/prepare_csv_benchmark.py resources/benchmarks/mp_20/train.csv mp_20_train_cin.tar.gz
python mat2seq/prepare_csv_benchmark.py resources/benchmarks/mp_20/val.csv mp_20_val_cin.tar.gz
python mat2seq/prepare_csv_benchmark.py resources/benchmarks/mp_20/test.csv mp_20_test_cin.tar.gz
```

Convert the .tar.gz files to .pkl.gz files for more efficient processing: 
```shell
python bin/tar_to_pickle.py mp_20_train_cin.tar.gz mp_20_train_cin.pkl.gz
python bin/tar_to_pickle.py mp_20_val_cin.tar.gz mp_20_val_cin.pkl.gz
python bin/tar_to_pickle.py mp_20_test_cin.tar.gz mp_20_test_cin.pkl.gz
```

Pre-process the benchmark CIF files:
```shell
python mat2seq/preprocess.py mp_20_train_cin.pkl.gz --out mp_20_train_cin_prep.pkl.gz --workers 4
python mat2seq/preprocess.py mp_20_val_cin.pkl.gz --out mp_20_val_cin_prep.pkl.gz --workers 4
python mat2seq/preprocess.py mp_20_test_cin.pkl.gz --out mp_20_test_cin_prep.pkl.gz --workers 4
```

### 2. Tokenize the pre-processed CIF files

Tokenize the benchmark training and validation sets:
```shell
python mat2seq/tokenize_cifs.py \
--train_fname mp_20_train_cin_prep.pkl.gz \
--val_fname mp_20_val_cin_prep.pkl.gz \
--out_dir tokens_mp_20/ \
--workers 4
```



### 3. Train the model

Train a LM model from scratch using only the benchmark training set:
```shell
python bin/train.py --config=config/crystallm_perov_5_small.yaml device=cuda dtype=float16
```

### 4. Generate CIF files

Generate the prompts from the CIF files of the test set:
```shell
python bin/make_prompts.py perov_5_test_prep.pkl.gz -o prompts_perov_5_test.tar.gz
```

Generate the CIF files, performing 20 generation attempts from each of the prompts: 
```shell
python mat2seq/generate_cifs.py --model ./out/cinllm_mp_20_314_pos_order --prompts prompts_mp20_pos.tar.gz --out gen_mp_20_pos_raw.tar.gz --device cuda --num-gens 1
```

### 5. Post-process the generated CIF files

Post-process the generated CIF files:
```shell
python bin/postprocess.py gen_perov_5_small_raw.tar.gz gen_perov_5_small.tar.gz
```

### 6. Compute the benchmark metrics

To compute the benchmark metrics using all available generations:
```shell
python bin/benchmark_metrics.py gen_perov_5_small.tar.gz perov_5_test_orig.tar.gz
```

To compute the benchmark metrics for the first generation attempt only (i.e. the _n=1_ case):
```shell
python bin/benchmark_metrics.py gen_perov_5_small.tar.gz perov_5_test_orig.tar.gz --num-gens 1
```

The `benchmark_metrics.py` script will process the given collection of CIF files, and compare them to the original CIF 
files. When all the processing is complete, the match rate and RMSE will be printed to the console.

The steps above can be performed with any of the other benchmark datasets as well, by simply substituting `perov_5` 
with `carbon_24`, `mp_20`, or `mpts_52`. For example, to apply this pipeline to Carbon-24, use 
`carbon_24_train_orig.tar.gz` in place of `perov_5_train_orig.tar.gz`, and so forth. The benchmark datasets are located 
in the [resources/benchmarks](resources/benchmarks) folder.

Since the benchmarking pipeline has previously been applied to all the benchmark datasets, we've uploaded the generated 
artifacts for convenience and reproducibility. All artifacts generated by the pipeline can therefore be downloaded 
directly using `bin/download.py`. For example, the tokenized Perov-5 dataset can be downloaded directly using
```shell
python bin/download.py tokens_perov_5.tar.gz
```
Also, the generated CIF files for a benchmark can be downloaded directly:
```shell
python bin/download.py gen_perov_5_small.tar.gz
```

All models come in two sizes: `small` and `large`. One exception is the model trained on the full 2.3M-compound dataset 
minus the MPTS-52 validation and training sets, `crystallm_v1_minus_mpts_52_small.tar.gz`, which is available in the 
small size only. Its generated CIF files can be found in `gen_v1_minus_mpts_52_small.tar.gz` (and the tokens are in 
`tokens_v1_minus_mpts_52.tar.gz`).

### Acknowledgments
K.Y. and S.J. acknowledge the support from U.S. National Science Foundation (NSF) grant MOMS2331036. First-principles calculations and structure optimization by K.A. were supported by the Center for Reconfigurable Electronic Materials Inspired by Nonlinear Dynamics (reMIND), an Energy Frontier Research Center funded by the Department of Energy under award DE-SC0023353. X.F.Q. acknowledges the support from the Air Force Office of Scientific Research (AFOSR) under Grant No. FA9550-24-1-0207 and NSF CMMI-2226908. This work was partially supported by the donors of ACSPetroleum Research Fund under Grant 65502-ND10. X.N.Q. and R.A. acknowledge the support from U.S. National Science Foundation (NSF) grant DMREF-2119103 and X.N.Q acknowledges the support from NSF through grants SHF-2215573, and IIS-2212419. Portions of this research were conducted with the advanced computing resources provided by Texas A&M High Performance Research Computing. M.Z. gratefully acknowledges the support of NIH R01-HD108794, NSF CAREER 2339524, US DoD FA8702-15-D-0001, ARPA-H BDF Toolbox, awards from Pfizer Research, Harvard Data Science Initiative, Amazon Faculty Research, Google Research Scholar Program, AstraZeneca Research, Roche Alliance with Distinguished Scientists, Sanofi iDEA-iTECH Award, Chan Zuckerberg Initiative, John and Virginia Kaneb Fellowship award at Harvard Medical School, Biswas Computational Biology Initiative in partnership with the Milken Institute, Kempner Institute for the Study of Natural and Artificial Intelligence, and Harvard Medical School Dean’s Innovation Fund. C.E. and H.J. acknowledge the support from the Molecule Maker Lab Institute: an AI research institute program supported by NSF under award No. 2019897 and No. 2034562. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

The code implementation is built upon [CrystaLLM repository](https://github.com/lantunes/CrystaLLM), and the implementation of CrystaLLM is built upon [nanoGPT repository](https://github.com/karpathy/nanoGPT).


## Citing Mat2Seq

Please use the following bibtex entry:
```
@inproceedings{yaninvariant,
  title={Invariant Tokenization of Crystalline Materials for Language Model Enabled Generation},
  author={Yan, Keqiang and Li, Xiner and Ling, Hongyi and Ashen, Kenna and Edwards, Carl and Arr{\'o}yave, Raymundo and Zitnik, Marinka and Ji, Heng and Qian, Xiaofeng and Qian, Xiaoning and others},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```
