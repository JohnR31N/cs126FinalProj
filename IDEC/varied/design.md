# Random geometric backgrounds and the reconstruction objective

This design was specified before fitting. The question is whether task-irrelevant geometric variation harms digit clustering, and whether changing the reconstruction target reduces that harm. Reuse the original grayscale MNIST input, architecture, 12,000/10,000 split, seeds 0--4, 100 pretraining epochs and 5,600 clustering updates. This experiment varies brightness in one channel.

## Data

Place two symbols in the same legal 5 by 5 boxes used by the earlier geometric study. Their bounding boxes do not overlap and have one pixel of clearance from all nonzero digit pixels. A data-only check found that some digits cannot accommodate three such boxes, so retain two for every image. Every stroke pixel and its clearance remain exact.

Construct a 96-template bank from four filled geometries (square, diamond, triangle, L), two radii (2.0 and 2.4 pixels), four angles (0, 20, 40, 60 degrees) and three intensities (0.6, 0.8, 1.0). Rasterize at eight samples per pixel axis before averaging. Choose templates with a balanced, seeded shuffle independently of digit labels. The assigned training backgrounds stay fixed across epochs. Training and held-out backgrounds use the same rules and independent RNG streams. Save two independently redrawn test backgrounds at the same sites for each digit.

Compare three domains. Clean images contain no symbol. Fixed-form images use the same circular template in both boxes. For each symbol, scale that circle to have the same sum of squared intensities as its matched varied template. Varied images use the 96-template bank. The two marked domains have the same locations, count and per-symbol squared energy. Their occupied pixel area and spatial pattern may differ. This control tests spatial variety at matched energy, and does not isolate every attribute separately.

## Training arms

Fit ordinary IDEC on clean images. For both background domains, compare full-image reconstruction throughout with a clean-digit reconstruction target throughout. All marked inputs stay identical between these target conditions. The clean target provides privileged information from the generator and acts as a mechanism diagnostic.

In the varied domain also cross the pretraining target and fine-tuning target independently. The two fine-tuning runs for one pretrained model share exactly its weights and K-means centers. The two stages can therefore be assessed separately. Include the previously selected random translation and Gaussian pretraining interventions, both with ordinary full-image fine-tuning, on these varied backgrounds. This yields nine arms and five seeds per arm. Match starting weights, minibatches, learning rate, latent dimension, cluster count and training budget.

## Measurements and interpretation

Fit cluster-to-digit names only on training predictions after fitting. Evaluate the same model on its in-domain test image, clean test digits and two independently redrawn backgrounds. Report digit ACC, paired assignment flips, the clean-to-marked accuracy gap and occupied clusters. Save all assignments and all per-seed results.

Measure decoder response to a test-background redraw immediately after pretraining and after fine-tuning. Project the reconstruction difference onto the actual pixel difference and divide by the squared norm of that pixel difference. Full tracking has score one, zero has no component along that change. Also record digit-pixel MSE and normalized error on the background symbols, which can show the reconstruction tradeoff. These readouts describe the decoder's use of information and do not establish absence of background information from the latent code.

The principal behavioral comparison is ordinary IDEC across clean and background training domains. The principal mechanism comparison holds varied inputs fixed and changes full to clean reconstruction targets. A reduction in background sensitivity together with improved digit accuracy supports a limitation of full-image reconstruction under these conditions. A target intervention that fails to improve clustering is evidence against that proposed remedy in this setting. Compare paired seed differences and their sample SD, preserve negative results, and keep claims local to this generator. All target, palette and geometry choices are fixed before fitting.
