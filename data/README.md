# Data

## Dataset
- **Name:** TrashNet, resized version (`dataset-resized.zip`)
- **Source:** Gary Thung and Mindy Yang, Stanford CS229 project (2016): https://github.com/garythung/trashnet
- **Content:** 2,527 photos (512×384) of single items on a white posterboard in 6 classes. We use 4 of them: **plastic (482), paper (594), glass (501), metal (410)**. Cardboard and trash are not used yet.
- **Licence:** the repository, including the data, is published under the MIT licence. We cite the authors in reports.
- **Personal data:** none. The photos show objects only.

## How the data gets here
The data is **downloaded, not committed** (`data/raw/` is in `.gitignore`).

`python src/baseline_pipeline.py` downloads the zip (~43 MB) and the official split lists on the first run, then unzips the images. Without internet access, download these three files from https://github.com/garythung/trashnet/tree/master/data and put them in `data/raw/`:

```
data/
  README.md
  raw/
    dataset-resized.zip
    one-indexed-files-notrash_train.txt
    one-indexed-files-notrash_test.txt
    dataset-resized/            # created by the pipeline when it unzips
      plastic/ paper/ glass/ metal/ cardboard/ trash/
```

## Preparation
None by hand. The pipeline unzips the images, checks that every listed image exists and is 512×384, and checks that each class has examples.
