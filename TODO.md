# Tasks

## Caching

 * Check if disabling caching from the command-line works
 * Revisit how llm_util does caching - as is using the cache bypasses echos

## Experimentation

 * Run experiments in sports_understanding
     * use smaller models till the task gets "interesting"?

## Learning methods

Start with distilling react traces.

In expt.py 
 * refactor by pulling out all the setup code in run() before the actual evaluation.
 * write a new command 'record' that records the results of an evaluation and saves 
   a json with all the measurements and the recording of each example.

Add a src/secretagent/learn subdirectory

Add a learn/distill_pot.py
 * takes every pot function for every example
 * uses simulated interfaces to convert to a 'canonical' functional form
   "def workflow(...) -> ..." 
 * uses ast to rename all the variables to v01, v02, ...
 * hashes them to get a smaller set of functions
 * computes coverage of each function (correct/incorrect)
 * does some sort of greedy set cover
 * incrementally calls a simulated interface to refactor the workflows
   into one program

## Known issues

 * Running the simulate_pydantic with tools leads to a bunch of
   litellm task warnings, which are meaningless but annoying and
   seemingly hard to fix.

