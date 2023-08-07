import argparse
from utils import *
from example_generator import *
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, default='text-davinci-003',
                        help='The name of the model to test')
    parser.add_argument('--questionnaire', required=True, type=str,
                        help='Comma-separated list of questionnaires')
    parser.add_argument('--shuffle-count', required=True, type=int, default=0,
                        help='Numbers of different orders. If set zero, run only the original order. If set n > 0, run the original order along with its n permutations. Defaults to zero.')
    parser.add_argument('--test-count', required=True, type=int, default=1,
                        help='Numbers of runs for a same order. Defaults to one.')
    parser.add_argument('--name-exp', type=str, default=None,
                        help='Name of this run. Is used to name the result files.')
    parser.add_argument('--significance-level', type=float, default=0.01,
                        help='The significance level for testing the difference of means between human and LLM. Defaults to 0.01.')
    parser.add_argument('--mode', type=str, default='auto',
                        help='For debugging.')

    # Generator-specific parameters, can be discarded if users implement their own generators
    parser.add_argument('--openai-organization', type=str, default='')
    parser.add_argument('--openai-key', type=str, default='')
    
    args = parser.parse_args()

    run_psychobench(args, example_generator)

