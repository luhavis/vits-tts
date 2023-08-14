
### Create conda env
```bash
conda create -n vits-tts python=3.10
```
### Activate conda env
```bash
activate vits-tts
```

### Install PyTorch
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Audio down sampling
- resampling 22,050 Hz
```bash
python data_downsample.py
```

### Cleaned train, validation script
```bash
python preprocess.py --text_index 2 --filelists configs/train.txt configs/train_val.txt --text_cleaners korean_cleaners
```

### Train
- audio data (22,050 Hz)
```bash
python train_ms.py -c {Config path} -m {Model name}
python train_ms.py -c configs/base_resample.json -m kss
```

### Tensorboard
```bash
tensorboard --logdir models/{Model name} --port 6006
```

### References
[jaywalnut310/vits](https://github.com/jaywalnut310/vits)

[CjangCjengh/vits](https://github.com/CjangCjengh/vits)

### Datasets
[KSS Dataset](https://huggingface.co/datasets/Bingsu/KSS_Dataset)


