# PsychoBench
**RESEARCH USE ONLY. NO COMMERCIAL USE ALLOWED**

Benchmarking LLMs' Psychological Portray.

## Usage
An example run:
```
python run_psychobench.py \
  --model gpt-3.5-turbo \
  --questionnaire Empathy \
  --shuffle-count 1 \
  --test-count 2
```

An example result:
| | Mean | STD | N |
| --- | --- | --- | --- |
| LLM | $µ_1$ = 6.05 | $s_1$ = 0.1732 | $n_1$ = 4 |
| Crowd | $µ_2$ = 4.92 | $s_2$ = 0.76 | $n_2$ = 112 |

## Argument Specification
1. `--questionnaire`: (Required) Select the questionnaire to run. For choises please see the list bellow.

2. `--model`: (Required) The name of the model to test.

3. `--shuffle-count`: (Required) Numbers of different orders. If set zero, run only the original order. If set n > 0, run the original order along with its n permutations. Defaults to zero.

4. `--test-count`: (Required) Numbers of runs for a same order. Defaults to one.

5. `--name-exp`: Name of this run. Is used to name the result files.

6. `--significance-level`: The significance level for testing the difference of means between human and LLM. Defaults to 0.01.

7. `--mode`: For debugging. To choose which part of the code is running.

Arguments related to `openai` API (can be discarded when users customize models):

1. `--openai-organization`: Your organization ID. Can be found in `Manage account -> Settings -> Organization ID`.

2. `--openai-key`: Your API key. Can be found in `View API keys -> API keys`.

## Benchmarking Your Own Model
It is easy! Just replace the function `example_generator` fed into the function `run_psychobench(args, generator)`.

Your customized function `your_generator()` does the following things:

1. Read questions from the file `--testing-file`. The file has the following format:
| Prompt: ... | order-1 | shuffle0-test0 | shuffle0-test1 | Prompt: ... | order-2 | shuffle0-test0 | shuffle0-test1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | 1 | | | Q3 | 3 | | |
| Q2 | 2 | | | Q5 | 5 | | |
| ... | ... | | | ... | ... | | |
| Qn | n | | | Q1 | 1 | | |

You can read the columns before each column starting with `order-`, which contains the shuffled questions for your input.

2. Call your own LLM and get the results.

3. Fill in the blank in the file `--testing-file`. **Remember**: No need to map the response to its original order. Our code will take care of it.

Please check `example_generator.py` for datailed information.

## Questionnaire List (Choices for Argument: --questionnaire)
1. Big Five Inventory: `--questionnaire BFI`

2. Dark Triad Dirty Dozen: `--questionnaire DTDD`

3. Eysenck Personality Questionnaire-Revised: `--questionnaire EPQ-R`

4. Experiences in Close Relationships-Revised (Adult Attachment Questionnaire): `--questionnaire ECR-R`

5. Vocational Interest Scale: `--questionnaire VIS`

6. General Self-Efficacy: `--questionnaire GSE`

7. Love of Money Scale: `--questionnaire LMS`

8. Bem's Sex Role Inventory: `--questionnaire BSRI`

9. Implicit Culture Belief: `--questionnaire ICB`

10. Revised Life Orientation Test: `--questionnaire LOT-R`

11. Empathy Scale: `--questionnaire Empathy`

12. Emotional Intelligence Scale: `--questionnaire EIS`

13. Wong and Law Emotional Intelligence Scale: `--questionnaire WLEIS`