import csv
import json
import os
import random
import scipy.stats as stats
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
import sys


def get_questionnaire(questionnaire_name):
    with open('questionnaires.json') as dataset:
        data = json.load(dataset)
    
    # Matching by questionnaire_name in dataset
    questionnaire = None
    for item in data:
        if item["name"] == questionnaire_name:
            questionnaire = item
    
    if questionnaire is None:
        print("Questionnaire not found.")
        sys.exit(1)

    return questionnaire


def plot_bar_chart(mean1_list, mean2_list, cat_list, save_name, title="Bar Chart", model="LLM"):
    # Plotting bar chart
    barWidth = 0.35
    br1 = np.arange(len(mean1_list))
    br2 = [x + barWidth for x in br1]
    figure_width = max(6, len(mean1_list) * 1.8)
    
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(figure_width, 6))

    # Plotting the bars
    ax.bar(br1, mean1_list, color='b', width=barWidth, alpha=0.5, label=model)
    ax.bar(br2, mean2_list, color='r', width=barWidth, alpha=0.5, label='Crowd')
    
    # Figure settings
    ax.set_title(title)
    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks([r + barWidth / 2 for r in range(len(br1))])
    ax.set_xticklabels(cat_list)
    ax.legend()
    plt.savefig(save_name, dpi=300)


def generation(questionnaire, args):
    test_count = args.test_count
    do_shuffle = args.shuffle_count
    output_file = args.testing_file
    csv_output = []
    questions_list = questionnaire["questions"] # get all questions

    for shuffle_count in range(do_shuffle + 1):
        question_indices = list(questions_list.keys())  # get the question indices

        # Shuffle the question indices
        if shuffle_count != 0:
            random.shuffle(question_indices)
        
        # Shuffle the questions order based on the shuffled indices
        questions = [f'{index}. {questions_list[question]}' for index, question in enumerate(question_indices, 1)]
        
        csv_output.append([f'Prompt: {questionnaire["prompt"]}'] + questions)
        csv_output.append([f'order-{shuffle_count+1}'] + question_indices)
        for count in range(test_count):
            csv_output.append([f'shuffle{shuffle_count}-test{count}'] + [''] * len(question_indices))

    csv_output = zip(*csv_output)
        
    # Write the csv file
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_output)


def convert_data(questionnaire, testing_file):
    # Check testing_file exist
    if not os.path.exists(testing_file):
        print("Testing file does not exist.")
        sys.exit(1)

    test_data = []
    
    with open(testing_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        
        # Take the index of column which refer to the question order
        order_indices = []
        for index, column in enumerate(header):
            if column.startswith("order"):
                order_indices.append(index)
                
        # For each question order, record the correspond test data
        for i in range(len(order_indices)):
            
            # start and end are the range of the test data which correspond to the current question order
            start = order_indices[i] + 1
            end = order_indices[i+1] - 1 if order_indices[i] != order_indices[-1] else len(header)
            
            # column index refer to the index of column within those test data
            for column_index in range(start, end):
                column_data = {}
                csvfile.seek(0)
                next(reader)
                
                # For each row in the table, take the question index x and related response y as `"x": y` format
                for row in reader:
                    try: 
                        # Check whether the question is a reverse scale
                        if int(row[start-1]) in questionnaire["reverse"]:
                            column_data[row[start-1]] = questionnaire["scale"] - int(row[column_index])
                        else:
                            column_data[row[start-1]] = int(row[column_index])
                    except ValueError:
                        print(f'Column {column_index + 1} has error.')
                        sys.exit(1)

                test_data.append(column_data)
            
    return test_data


def analysis_results(questionnaire, args):
    significance_level = args.significance_level
    testing_file = args.testing_file
    result_file = args.results_file
    model = args.model
    test_data = convert_data(questionnaire, testing_file)
    mean1_list = []
    mean2_list = []
    cat_list = []
    
    output_text = ''
    # Analysis by each category
    for cat in questionnaire['categories']:
        cat_list.append(cat['cat_name'])
        result_list = []
        
        for test in test_data:
            results = []
            
            for key in test:
                if int(key) in cat['cat_questions']:
                    results.append(test[key])
            
            # Getting the computation mode (SUM or AVG)
            if "compute_mode" in cat and cat["compute_mode"] == "SUM":
                result_list.append(sum(results))
            else:
                result_list.append(mean(results))
                
        if len(result_list) < 2:
            raise ValueError("result_list should have at least 2 elements.")
            
            
        # Collecting LLM's data
        epsilon = 1e-8  # Setting epsilon to avoid zero standard deviation
        mean1, std1, n1 = mean(result_list), stdev(result_list) + epsilon, len(result_list)
        mean1_4sf = float(f'{mean1:.4g}')
        std1_4sf = float(f'{std1:.4g}')
        output_text += f'# Questionnaire: {questionnaire["name"]}\n\n'
        output_text += f'## Category: {cat["cat_name"]}\n\n'
        output_text += f'| | Mean | STD | N |\n'
        output_text += f'| --- | --- | --- | --- |\n'
        output_text += f'| LLM | $µ_1$ = {mean1_4sf} | $s_1$ = {std1_4sf} | $n_1$ = {n1} |\n'
        mean1_list.append(mean1)
        
        # Performing F-test and T-test for each category
        for index, crowd in enumerate(cat["crowd"]):
            if "crowd_name" in crowd and crowd["crowd_name"] is not None:
                output_text += f'-------------\n{crowd["crowd_name"]}\n'
            
            # Collecting Crowd data
            mean2, std2, n2 = crowd["crowd_sample_mean"], crowd["crowd_sample_sd"], crowd["crowd_sample_num"]
            output_text += f'| Crowd | $µ_2$ = {mean2} | $s_2$ = {std2} | $n_2$ = {n2} |\n'
            mean2_list.append(mean2)


            # Performing F-test
            output_text += '\n### F-Test:\n\n'
            
            if std1 > std2:
                f_value = std1 ** 2 / std2 ** 2
                df1, df2 = n1 - 1, n2 - 1
            else:
                f_value = std2 ** 2 / std1 ** 2
                df1, df2 = n2 - 1, n1 - 1

            p_value = (1 - stats.f.cdf(f_value, df1, df2)) * 2
            equal_var = True if p_value > significance_level / 2 else False
            
            f_value = float(f'{f_value:.4g}')
            p_value = float(f'{p_value:.4g}')
            output_text += f'\tf-value = {f_value} ($df_1$ = {df1}, $df_2$ = {df2})\n\n'
            output_text += f'\tp-value = {p_value} (two-tailed test)\n\n'
            output_text += '\tNull hypothesis $H_0$ ($s_1^2$ = $s_2^2$): '

            if p_value > significance_level:
                output_text += f'Since p-value ({p_value}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n'
                output_text += f'\tConclusion: ($s_1^2$ = $s_2^2$) The variance of LLM\'s average responses is statistically equal to the variance of crowd average.\n\n'
            else:
                output_text += f'Since p-value ({p_value}) < α ({significance_level}), $H_0$ is rejected.\n\n'
                output_text += f'\tConclusion: ($s_1^2$ ≠ $s_2^2$) The variance of LLM\'s average responses is statistically unequal to the variance of crowd average.\n\n'


            # Performing T-test
            output_text += '### Two Sample T-Test (Equal Variance):\n\n' if equal_var else '### Two Sample T-test (Welch\'s T-Test):\n\n'
            
            df = n1 + n2 - 2 if equal_var else ((std1**2 / n1 + std2**2 / n2)**2) / ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1))
            t_value, p_value = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=equal_var)
            df = float(f'{df:.4g}')

            t_value = float(f'{t_value:.4g}')
            p_value = float(f'{p_value:.4g}')
            output_text += f'\tt-value = {t_value} (d.f. = {df})\n\n'
            output_text += f'\tp-value = {p_value} (two-tailed test)\n\n'
            
            output_text += '\tNull hypothesis $H_0$ ($µ_1$ = $µ_2$): '
            if p_value > significance_level:
                output_text += f'Since p-value ({p_value}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n'
                output_text += f'\tConclusion: ($µ_1$ = $µ_2$) The average of LLM\'s responses is assumed to be equal to the average of crowd data.\n\n'
            else:
                output_text += f'Since p-value ({p_value}) < α ({significance_level}), $H_0$ is rejected.\n\n'
                if t_value > 0:
                    output_text += '\tAlternative hypothesis $H_1$ ($µ_1$ > $µ_2$): '
                    output_text += f'Since p-value ({1-p_value/2}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n'
                    output_text += f'\tConclusion: ($µ_1$ > $µ_2$) The average of LLM\'s responses is assumed to be larger than the average of crowd data.\n\n'
                else:
                    output_text += '\tAlternative hypothesis $H_1$ ($µ_1$ < $µ_2$): '
                    output_text += f'Since p-value ({1-p_value/2}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n'
                    output_text += f'\tConclusion: ($µ_1$ < $µ_2$) The average of LLM\'s responses is assumed to be smaller than the average of crowd data.\n\n'
        
        output_text += f'![Bar Chart]({args.figures_file} "Bar Chart of {args.model} on {questionnaire["name"]}")\n'
    
    # Writing the results into a text file
    with open(result_file, "w") as f:
        f.write(output_text)
    
    plot_bar_chart(mean1_list, mean2_list, cat_list, save_name=args.figures_file, title=questionnaire["name"], model=model)


def run_psychobench(args, generator):
    # Get questionnaire
    questionnaire = get_questionnaire(args.questionnaire)
    args.testing_file = f'results/{args.name_exp}.csv' if args.name_exp is not None else f'results/{args.model}-{questionnaire["name"]}.csv'
    args.results_file = f'results/{args.name_exp}.txt' if args.name_exp is not None else f'results/{args.model}-{questionnaire["name"]}.md'
    args.figures_file = f'{args.name_exp}.png' if args.name_exp is not None else f'{args.model}-{questionnaire["name"]}.png'
    
    # Generation
    if args.mode in ['generation', 'auto']:
        generation(questionnaire, args)
    
    # Testing
    if args.mode in ['testing', 'auto']:
        generator(questionnaire, args)
        
    # Analysis
    if args.mode in ['analysis', 'auto']:
        analysis_results(questionnaire, args)

