# Project decisions

## Lab 03 — Acceptance

### Decision after the baseline
Decision: **Narrow**

Reason: The simple colour-based baseline reached a macro F1 of 0.607 on 332 held-out TrashNet photos. Glass was the weakest class, with a recall of 0.50 (41 of 82). An important limitation is that TrashNet shows only one clean item on a white board. We therefore decided to narrow the operating conditions: the system handles **one item at a time, placed on a plain tray under a fixed camera**. It does not handle a pile of mixed waste or a full bin. The users, the four materials and the useful output stay the same as in the project brief. The baseline shows that there is a learnable signal even without deep learning, so the direction is worth continuing within these conditions.

### Required behavior
- **Normal input:** one item on the tray, sharp and well-lit photo. The system returns the material (plastic, paper, glass or metal) and a confidence score. Staff put the item into that container.
- **Invalid input:** empty tray, more than one item, item partly outside the frame, or a photo that is too blurry or too dark. The system gives no material. Instead it shows the reason, e.g. "Place one item in the middle of the tray", and the staff member retakes the photo.
- **Valid input with an uncertain result:** the top confidence is below the cutoff, or the top two materials are close. The system shows "Check by hand" together with the two most likely materials. A staff member decides, and the photo and their decision are saved for later labelling.

### One quality that matters
- **Response time:** staff should not wait at the tray. Our target is a result within 2 seconds from photo to screen on a normal laptop CPU without a GPU. This is not measured yet. First we need to time the full path (photo → checks → model → screen) once a model is chosen.

### One acceptance check
- **Important failure:** a glass item is sent to the paper, plastic or metal container. Broken glass is dangerous for people who sort by hand, and it can make a whole batch of paper unusable for recycling.
- **Metric that would show this failure:** **glass recall** on the evaluation set, i.e. the share of glass items that are either labelled glass or sent to "Check by hand". This is the same as 1 − glass miss rate. It is taken from the per-class recall that the Lab 02 pipeline already saves.
- **Measured baseline (Lab 02, not a target):** glass recall = **0.50**. 41 of 82 glass test images were correct: 14 were predicted as plastic, 11 as paper and 16 as metal.
- **Preliminary acceptance threshold:** glass recall **≥ 0.90** in a later evaluation, with "Check by hand" counted as caught.
- **Where will we check it?** First on TrashNet's held-out test list (82 glass images). Later on our own labelled photos from a real collection point, if we get permission.
- **Why this number for now?** At a small collection point, staff can catch an occasional mistake, but missing half of the glass is not acceptable. 0.90 is a starting point the team agreed on, not an industry standard. It may become stricter once we know the real tolerance.
- **Evidence still missing:**
  - How much wrong-stream glass the recycling buyer actually accepts.
  - Glass recall on real, dirty, crushed items from a collection point.
  - A limit on how many items may go to "Check by hand": **not set yet**. Without one, the system could reach 0.90 just by sending everything to a person. We first need to count how many items per hour staff can check by hand.

### Failure cases
| Area | What could go wrong and why it matters | How would we notice it? | Planned response |
| --- | --- | --- | --- |
| Data or input | Two items on the tray, or an item half outside the photo. The system gives one material for both, so one item goes to the wrong container. | Input check finds more than one object or an object touching the image edge; staff report "wrong item" corrections. | Reject the photo with "Place one item in the middle of the tray" and ask for a retake. |
| Model behavior | A clear glass bottle is predicted as plastic, because both are transparent and bottle-shaped. Glass ends up in the plastic stream. **(Matches the acceptance check.)** | Glass recall on the evaluation set and the glass→plastic cell of the confusion matrix (14 of 82 in the baseline); staff corrections logged as "glass → plastic". | Send low-confidence glass/plastic results to "Check by hand"; add more clear-glass training photos; re-check glass recall against 0.90. |
| Integration or workflow | Too many items go to "Check by hand" at busy times. Staff start skipping the check and guess, so the fallback stops protecting against mistakes. | Share of items sent to "Check by hand" per hour, and how long those items wait before a decision is logged. | Show the two most likely materials to make checking fast; review the uncertainty cutoff weekly; set a limit once we have staff capacity numbers. |
