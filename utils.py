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
    try:
        with open('questionnaires.json') as dataset:
            data = json.load(dataset)
    except FileNotFoundError:
        raise FileNotFoundError("The 'questionnaires.json' file does not exist.")

    # Matching by questionnaire_name in dataset
    questionnaire = None
    for item in data:
        if item["name"] == questionnaire_name:
            questionnaire = item

    if questionnaire is None:
        raise ValueError("Questionnaire not found.")

    return questionnaire



def plot_bar_chart(value_list, cat_list, item_list, save_name, title="Bar Chart"):
    num_bars = len(value_list)
    bar_width = 1 / num_bars * 0.8
    figure_width = max(8, len(cat_list) * 1.2)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(figure_width, 8))

    # Plotting the bars
    colors = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'w']
    br = [np.arange(len(cat_list)) + x * bar_width for x in range(num_bars)]
    for i, values in enumerate(value_list):
        ax.bar(br[i], values, color=colors[i % len(colors)], width=bar_width, alpha=0.5, label=item_list[i])

    # Figure settings
    ax.set_title(title)
    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks([r + bar_width * (num_bars - 1) / 2 for r in range(len(cat_list))])
    if title in ['CABIN']:
        ax.set_xticklabels(cat_list, rotation=20, ha='right')
    else:
        ax.set_xticklabels(cat_list)
    ax.legend()
    plt.savefig(f'results/figures/{save_name}', dpi=300)



def generate_testfile(questionnaire, args):
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
        csv_output.append([f'order-{shuffle_count}'] + question_indices)
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
                        if row[start-1] in questionnaire["reverse"]:
                            column_data[int(row[start-1])] = questionnaire["scale"] - int(row[column_index])
                        else:
                            column_data[int(row[start-1])] = int(row[column_index])
                    except ValueError:
                        print(f'Column {column_index + 1} has error.')
                        sys.exit(1)

                test_data.append(column_data)
            
    return test_data



def compute_statistics(questionnaire, data_list):
    results = []
    
    for cat in questionnaire["categories"]:
        scores_list = []
        
        for data in data_list:
            scores = []
            for key in data:
                if key in cat["cat_questions"]:
                    scores.append(data[key])
            
            # Getting the computation mode (SUM or AVG)
            if questionnaire["compute_mode"] == "SUM":
                scores_list.append(sum(scores))
            else:
                scores_list.append(mean(scores))
        
        if len(scores_list) < 2:
            raise ValueError("The test file should have at least 2 test cases.")
        
        results.append((mean(scores_list), stdev(scores_list), len(scores_list)))
        
    return results



def hypothesis_testing(result1, result2, significance_level, model, crowd_name):
    output_list = ''
    output_text = f'### Compare with {crowd_name}\n'

    # Extract the mean, std and size for both data sets
    mean1, std1, n1 = result1
    mean2, std2, n2 = result2
    output_list += f'{mean2:.1f} $\pm$ {std2:.1f}'
    
    # Add an epsilon to prevent the zero standard deviarion
    epsilon = 1e-8
    std1 += epsilon
    std2 += epsilon
    
    output_text += '\n- **Statistic**:\n'
    output_text += f'{model}:\tmean1 = {mean1:.1f},\tstd1 = {std1:.1f},\tn1 = {n1}\n'
    output_text += f'{crowd_name}:\tmean2 = {mean2:.1f},\tstd2 = {std2:.1f},\tn2 = {n2}\n'
    
    # Perform F-test
    output_text += '\n- **F-Test:**\n\n'
    
    if std1 > std2:
        f_value = std1 ** 2 / std2 ** 2
        df1, df2 = n1 - 1, n2 - 1
    else:
        f_value = std2 ** 2 / std1 ** 2
        df1, df2 = n2 - 1, n1 - 1

    p_value = (1 - stats.f.cdf(f_value, df1, df2)) * 2
    equal_var = True if p_value > significance_level else False
    
    output_text += f'\tf-value = {f_value:.4f}\t($df_1$ = {df1}, $df_2$ = {df2})\n\n'
    output_text += f'\tp-value = {p_value:.4f}\t(two-tailed test)\n\n'
    output_text += '\tNull hypothesis $H_0$ ($s_1^2$ = $s_2^2$): '

    if p_value > significance_level:
        output_text += f'\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n'
        output_text += f'\t**Conclusion ($s_1^2$ = $s_2^2$):** The variance of average scores responsed by {model} is statistically equal to that responsed by {crowd_name} in this category.\n\n'
    else:
        output_text += f'\tSince p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n'
        output_text += f'\t**Conclusion ($s_1^2$ ≠ $s_2^2$):** The variance of average scores responsed by {model} is statistically unequal to that responsed by {crowd_name} in this category.\n\n'

    # Performing T-test
    output_text += '- **Two Sample T-Test (Equal Variance):**\n\n' if equal_var else '- **Two Sample T-test (Welch\'s T-Test):**\n\n'
    
    df = n1 + n2 - 2 if equal_var else ((std1**2 / n1 + std2**2 / n2)**2) / ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1))
    t_value, p_value = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=equal_var)

    output_text += f'\tt-value = {t_value:.4f}\t($df$ = {df:.1f})\n\n'
    output_text += f'\tp-value = {p_value:.4f}\t(two-tailed test)\n\n'
    
    output_text += '\tNull hypothesis $H_0$ ($µ_1$ = $µ_2$): '
    if p_value > significance_level:
        output_text += f'\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n'
        output_text += f'\t**Conclusion ($µ_1$ = $µ_2$):** The average scores of {model} is assumed to be equal to the average scores of {crowd_name} in this category.\n\n'
        # output_list += f' ( $-$ )'

    else:
        output_text += f'Since p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n'
        if t_value > 0:
            output_text += '\tAlternative hypothesis $H_1$ ($µ_1$ > $µ_2$): '
            output_text += f'\tSince p-value ({(1-p_value/2):.1f}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n'
            output_text += f'\t**Conclusion ($µ_1$ > $µ_2$):** The average scores of {model} is assumed to be larger than the average scores of {crowd_name} in this category.\n\n'
            # output_list += f' ( $\\uparrow$ )'
        else:
            output_text += '\tAlternative hypothesis $H_1$ ($µ_1$ < $µ_2$): '
            output_text += f'\tSince p-value ({(1-p_value/2):.1f}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n'
            output_text += f'\t**Conclusion ($µ_1$ < $µ_2$):** The average scores of {model} is assumed to be smaller than the average scores of {crowd_name} in this category.\n\n'
            # output_list += f' ( $\\downarrow$ )'

    output_list += f' | '
    return (output_text, output_list)



def analysis_results(questionnaire, args):
    significance_level = args.significance_level
    testing_file = args.testing_file
    result_file = args.results_file
    model = args.model
    
    test_data = convert_data(questionnaire, testing_file)
    test_results = compute_statistics(questionnaire, test_data)
    cat_list = [cat['cat_name'] for cat in questionnaire['categories']]
    crowd_list = [(c["crowd_name"], c["n"]) for c in questionnaire['categories'][0]["crowd"]]
    mean_list = [[] for i in range(len(crowd_list) + 1)]
    
    output_list = f'# {questionnaire["name"]} Results\n\n'
    output_list += f'| Category | {model} (n = {len(test_data)}) | ' + ' | '.join([f'{c[0]} (n = {c[1]})' for c in crowd_list]) + ' |\n'
    output_list += '| :---: | ' + ' | '.join([":---:" for i in range(len(crowd_list) + 1)]) + ' |\n'
    output_text = ''

    # Analysis by each category
    for cat_index, cat in enumerate(questionnaire['categories']):
        output_text += f'## {cat["cat_name"]}\n'
        output_list += f'| {cat["cat_name"]} | {test_results[cat_index][0]:.1f} $\pm$ {test_results[cat_index][1]:.1f} | '
        mean_list[0].append(test_results[cat_index][0])
        
        for crowd_index, crowd_group in enumerate(crowd_list):
            crowd_data = (cat["crowd"][crowd_index]["mean"], cat["crowd"][crowd_index]["std"], cat["crowd"][crowd_index]["n"])
            result_text, result_list = hypothesis_testing(test_results[cat_index], crowd_data, significance_level, model, crowd_group[0])
            output_list += result_list
            output_text += result_text
            mean_list[crowd_index+1].append(crowd_data[0])
            
        output_list += '\n'
    
    plot_bar_chart(mean_list, cat_list, [model] + [c[0] for c in crowd_list], save_name=args.figures_file, title=questionnaire["name"])
    output_list += f'\n\n![Bar Chart](figures/{args.figures_file} "Bar Chart of {model} on {questionnaire["name"]}")\n\n'
    
    # Writing the results into a text file
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(output_list + output_text)



def run_psychobench(args, generator):
    
    # Extract the targeted questionnaires
    questionnaire_list = ['BFI', 'DTDD', 'EPQ-R', 'ECR-R', 'CABIN', 'GSE', 'LMS', 'BSRI', 'ICB', 'LOT-R', 'Empathy', 'EIS', 'WLEIS'] \
                         if args.questionnaire == 'ALL' else args.questionnaire.split(',')
    
    for questionnaire_name in questionnaire_list:
        # Get questionnaire
        questionnaire = get_questionnaire(questionnaire_name)
        args.testing_file = f'results/{args.name_exp}-{questionnaire["name"]}.csv' if args.name_exp is not None else f'results/{args.model}-{questionnaire["name"]}.csv'
        args.results_file = f'results/{args.name_exp}-{questionnaire["name"]}.md' if args.name_exp is not None else f'results/{args.model}-{questionnaire["name"]}.md'
        args.figures_file = f'{args.name_exp}-{questionnaire["name"]}.png' if args.name_exp is not None else f'{args.model}-{questionnaire["name"]}.png'

        os.makedirs("results", exist_ok=True)
        os.makedirs("results/figures", exist_ok=True)
        
        # Generation
        if args.mode in ['generation', 'auto']:
            generate_testfile(questionnaire, args)
        
        # Testing
        if args.mode in ['testing', 'auto']:
            generator(questionnaire, args)
            
        # Analysis
        if args.mode in ['analysis', 'auto']:
            try:
                analysis_results(questionnaire, args)
            except:
                print(f'Unable to analysis {args.testing_file}.')

