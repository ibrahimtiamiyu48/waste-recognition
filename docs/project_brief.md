# Project brief

## User and situation
Staff at a recycling point, for example a campus or dormitory collection station, have to sort what people throw away. People often put items in the wrong container. Staff then re-sort by hand, or the batch is contaminated, so a paper bale with glass in it may be rejected. We want each item to be sorted into the right material stream more often, with less manual re-checking.

## What the system should do
- Input: one photo of a single waste item, taken at the collection point (phone or a fixed camera above a tray).
- Useful output: the item's material (**plastic, paper, glass or metal**) with a confidence score.
- Action or decision after the output: the item goes to the matching container. When the system is unsure, a staff member checks the item and decides.

## Why AI may help
The same material looks very different from item to item: clear vs. green glass, crushed cans vs. foil, glossy vs. plain paper, all kinds of plastic packaging. Fixed rules on colour or shape break quickly, so a model can learn these visual patterns from labelled photos. Besides the model, the system also needs:
- an input check (is there exactly one item, is the photo sharp and well lit),
- a confidence cutoff with a "check by hand" path for uncertain items,
- a simple screen or indicator that shows staff which container to use,
- staff who handle unsure items and can correct wrong results.

## Initial data plan
- Where the data may come from: TrashNet (Thung & Yang, Stanford CS229, 2016), with about 2,500 labelled photos of glass, paper, cardboard, plastic, metal and other trash. Later, possibly RealWaste (UCI) or TACO for more realistic, messy scenes. Photos we take ourselves at a campus collection point could also help.
- What we can access now: TrashNet's resized images (512×384) and its train/val/test file lists, downloadable from the authors' GitHub repository under the MIT licence.
- What still needs confirmation:
  - Is a single item on a white background close enough to real use, or do we need our own photos from an actual collection point?
  - Should cardboard count as paper? Recycling points usually collect them together. For now we leave cardboard out.
  - Are we allowed to take photos at a campus collection point, and who would label them?

## Three next actions
1. Download TrashNet, check the image format, class counts and the official split lists.
2. Ask the campus facilities office whether we may photograph items at a collection point.
3. Make sure every team member can clone the repository and run the first script.
