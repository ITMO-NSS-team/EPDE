# Fitted angular frequencies

Verified equivalent systems count as successful discoveries. A Lorenz candidate qualifies when its equations together span all three laws.

Run `python run.py` from this directory in the EPDE environment. The runner resumes completed jobs from their score files. It schedules one worker, binds it to one logical CPU, and limits BLAS, OpenMP and PyTorch to one thread. The default host CPU ceiling is 75%. The runner reserves one worker's share before launching or resuming it and suspends its own worker when a load sample exceeds the ceiling. Use `--cpu-ceiling` to change that percentage.

The screen pairs seed 923701 at 0% and 5% noise across the 15 systems and the harmonic control. The native arm uses the existing frequency grid. The FFT arm allows standalone trig terms and sine–cosine products on that grid. The fitted arm uses the same grammar and fits an independent continuous angular frequency for each factor. All arms use the blind version of forced KdV. The search receives field measurements and coordinates. Its scoring registry describes the forcing as two generic factors.

Each scalar search runs 50 epochs and each coupled search runs 150, with patience disabled. Each paired native run sets the work budget through its number of fitness calls plus extra coefficient solves. The comparison records the native archive available at that budget and the final native archive. It records complete-system recovery both with the registered 1% frequency tolerance and with the frequency values ignored. The generated-system readouts use actual system handovers. Clean validation replays the training record. At 5% noise, validation evaluates frozen frequencies and coefficients on a second noise draw on the same grid.

The highest strict matched-budget recovery count on the 15 systems advances if it exceeds native recovery. Lower total measured work breaks a tie between improved arms. The runner compares the selected arm with native on seeds 923702 and 923703. Per-job measurements are in `results/`. The screen writes `selection.json` and the completed campaign writes `results.json`.
