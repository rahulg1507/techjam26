# Data

This folder is gitignored — datasets are too large to commit and are publicly downloadable.

## Expected structure
```
data/
├── cifake/
│   ├── train/
│   │   ├── REAL/
│   │   └── FAKE/
│   └── test/
│       ├── REAL/
│       └── FAKE/
├── sid_set/
├── wildfake/
└── validation/          # demo-only, not for training
    ├── coco_val2017/    # real
    └── dalle_advanced/  # AIGC
```

## Download instructions

**CIFAKE** (Kaggle — needs Kaggle account/API key):
```bash
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/cifake --unzip
```

**SID_Set** (Hugging Face):
```bash
huggingface-cli download saberzl/SID_Set --repo-type dataset --local-dir data/sid_set
```

**WildFake** (ModelScope — translate page via translate button before browsing):
See https://modelscope.cn/datasets/hy2628982280/WildFake/summary

**Validation set (demo only — do NOT train on this):**
- COCO val2017: https://cocodataset.org/#download
- DALL·E Advanced subset: per WildFake breakdown
