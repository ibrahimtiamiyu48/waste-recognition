# Split strategy

**Branch: model trained by our team.**

- **Training:** TrashNet's own training list (`one-indexed-files-notrash_train.txt`), filtered to our 4 materials: 1,390 images.
- **Evaluation:** TrashNet's own test list, filtered the same way: 332 images (plastic 74, paper 108, glass 82, metal 68). These images are never used to fit, choose or adjust the baseline. The validation list is not used in this first run. We keep it for tuning in later Labs, so the test images stay untouched.
- **Why this split:** we use the authors' fixed lists, not our own random split, so every run and every teammate evaluates on exactly the same images. The pipeline also checks that no image is in both lists.
- **Does it look like real future input?** No, not really. TrashNet shows one clean item on a white board in good light. At a real collection point, items are dirty, crushed, partly covered and photographed on busy backgrounds. This is a limitation.
- **One limitation of this evaluation:** some objects were probably photographed more than once from different angles. If the same object appears in both training and test, the result may look better than it is. The dataset does not record object IDs, so we cannot check this yet.
