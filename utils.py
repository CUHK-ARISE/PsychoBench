import csv
import json
import os
import random
import scipy.stats as stats
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


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
                        if int(row[start-1]) in questionnaire["reverse"]:
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


import json 
import copy
import requests

payload_template = {
    "questions": [
        {"text": "You regularly make new friends.", "answer": None},
        {"text": "You spend a lot of your free time exploring various random topics that pique your interest.", "answer": None},
        {"text": "Seeing other people cry can easily make you feel like you want to cry too.", "answer": None},
        {"text": "You often make a backup plan for a backup plan.", "answer": None},
        {"text": "You usually stay calm, even under a lot of pressure.", "answer": None},
        {"text": "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know.", "answer": None},
        {"text": "You prefer to completely finish one project before starting another.", "answer": None},
        {"text": "You are very sentimental.", "answer": None},
        {"text": "You like to use organizing tools like schedules and lists.", "answer": None},
        {"text": "Even a small mistake can cause you to doubt your overall abilities and knowledge.", "answer": None},
        {"text": "You feel comfortable just walking up to someone you find interesting and striking up a conversation.", "answer": None},
        {"text": "You are not too interested in discussing various interpretations and analyses of creative works.", "answer": None},
        {"text": "You are more inclined to follow your head than your heart.", "answer": None},
        {"text": "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.", "answer": None},
        {"text": "You rarely worry about whether you make a good impression on people you meet.", "answer": None},
        {"text": "You enjoy participating in group activities.", "answer": None},
        {"text": "You like books and movies that make you come up with your own interpretation of the ending.", "answer": None},
        {"text": "Your happiness comes more from helping others accomplish things than your own accomplishments.", "answer": None},
        {"text": "You are interested in so many things that you find it difficult to choose what to try next.", "answer": None},
        {"text": "You are prone to worrying that things will take a turn for the worse.", "answer": None},
        {"text": "You avoid leadership roles in group settings.", "answer": None},
        {"text": "You are definitely not an artistic type of person.", "answer": None},
        {"text": "You think the world would be a better place if people relied more on rationality and less on their feelings.", "answer": None},
        {"text": "You prefer to do your chores before allowing yourself to relax.", "answer": None},
        {"text": "You enjoy watching people argue.", "answer": None},
        {"text": "You tend to avoid drawing attention to yourself.", "answer": None},
        {"text": "Your mood can change very quickly.", "answer": None},
        {"text": "You lose patience with people who are not as efficient as you.", "answer": None},
        {"text": "You often end up doing things at the last possible moment.", "answer": None},
        {"text": "You have always been fascinated by the question of what, if anything, happens after death.", "answer": None},
        {"text": "You usually prefer to be around others rather than on your own.", "answer": None},
        {"text": "You become bored or lose interest when the discussion gets highly theoretical.", "answer": None},
        {"text": "You find it easy to empathize with a person whose experiences are very different from yours.", "answer": None},
        {"text": "You usually postpone finalizing decisions for as long as possible.", "answer": None},
        {"text": "You rarely second-guess the choices that you have made.", "answer": None},
        {"text": "After a long and exhausting week, a lively social event is just what you need.", "answer": None},
        {"text": "You enjoy going to art museums.", "answer": None},
        {"text": "You often have a hard time understanding other people’s feelings.", "answer": None},
        {"text": "You like to have a to-do list for each day.", "answer": None},
        {"text": "You rarely feel insecure.", "answer": None},
        {"text": "You avoid making phone calls.", "answer": None},
        {"text": "You often spend a lot of time trying to understand views that are very different from your own.", "answer": None},
        {"text": "In your social circle, you are often the one who contacts your friends and initiates activities.", "answer": None},
        {"text": "If your plans are interrupted, your top priority is to get back on track as soon as possible.", "answer": None},
        {"text": "You are still bothered by mistakes that you made a long time ago.", "answer": None},
        {"text": "You rarely contemplate the reasons for human existence or the meaning of life.", "answer": None},
        {"text": "Your emotions control you more than you control them.", "answer": None},
        {"text": "You take great care not to make people look bad, even when it is completely their fault.", "answer": None},
        {"text": "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.", "answer": None},
        {"text": "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.", "answer": None},
        {"text": "You would love a job that requires you to work alone most of the time.", "answer": None},
        {"text": "You believe that pondering abstract philosophical questions is a waste of time.", "answer": None},
        {"text": "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.", "answer": None},
        {"text": "You know at first glance how someone is feeling.", "answer": None},
        {"text": "You often feel overwhelmed.", "answer": None},
        {"text": "You complete things methodically without skipping over any steps.", "answer": None},
        {"text": "You are very intrigued by things labeled as controversial.", "answer": None},
        {"text": "You would pass along a good opportunity if you thought someone else needed it more.", "answer": None},
        {"text": "You struggle with deadlines.", "answer": None},
        {"text": "You feel confident that things will work out for you.", "answer": None}
    ],
    "gender": None,
    "inviteCode": "",
    "teamInviteKey": "",
    "extraData": []
}

role_mapping = {'ISTJ': 'Logistician', 'ISTP': 'Virtuoso', 'ISFJ': 'Defender', 'ISFP': 'Adventurer', 'INFJ': 'Advocate', 'INFP': 'Mediator', 'INTJ': 'Architect', 'INTP': 'Logician', 'ESTP': 'Entrepreneur', 'ESTJ': 'Executive', 'ESFP': 'Entertainer', 'ESFJ': 'Consul', 'ENFP': 'Campaigner', 'ENFJ': 'Protagonist', 'ENTP': 'Debater', 'ENTJ': 'Commander'}


def parsing(score_list):
    code = ''
    
    if score_list[0] >= 50:
        code = code + 'E'
    else:
        code = code + 'I'

    if score_list[1] >= 50:
        code = code + 'N'
    else:
        code = code + 'S'

    if score_list[2] >= 50:
        code = code + 'T'
    else:
        code = code + 'F'

    if score_list[3] >= 50:
        code = code + 'J'
    else:
        code = code + 'P'

    if score_list[4] >= 50:
        code = code + '-A'
    else:
        code = code + '-T'

    return code, role_mapping[code[:4]]


# scores: List of int, length: 60, int range: -3~3
def query_16personalities_api(scores):
    payload = copy.deepcopy(payload_template)
    
    for index, score in enumerate(scores):
        payload['questions'][index]["answer"] = score
    
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "content-length": "5708",
        "content-type": "application/json",
        "origin": "https://www.16personalities.com",
        "referer": "https://www.16personalities.com/free-personality-test",
        "sec-ch-ua": "'Not_A Brand';v='99', 'Google Chrome';v='109', 'Chromium';v='109'",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
    }
    
    session = requests.session()
    r = session.post('https://www.16personalities.com/test-results', data=json.dumps(payload), headers=headers)
    
    sess_r = session.get("https://www.16personalities.com/api/session")
    scores = sess_r.json()['user']['scores']
    
    if sess_r.json()['user']['traits']['energy'] != 'Extraverted':
        energy_value = 100 - (101 + scores[0]) // 2
    else:
        energy_value = (101 + scores[0]) // 2
    if sess_r.json()['user']['traits']['mind'] != 'Intuitive':
        mind_value = 100 - (101 + scores[1]) // 2
    else:
        mind_value = (101 + scores[1]) // 2
    if sess_r.json()['user']['traits']['nature'] != 'Thinking':
        nature_value = 100 - (101 + scores[2]) // 2
    else:
        nature_value = (101 + scores[2]) // 2
    if sess_r.json()['user']['traits']['tactics'] != 'Judging':
        tactics_value = 100 - (101 + scores[3]) // 2
    else:
        tactics_value = (101 + scores[3]) // 2
    if sess_r.json()['user']['traits']['identity'] != 'Assertive':
        identity_value = 100 - (101 + scores[4]) // 2
    else:
        identity_value = (101 + scores[4]) // 2
    
    code, role = parsing([energy_value, mind_value, nature_value, tactics_value, identity_value])
    
    return code, role, [energy_value, mind_value, nature_value, tactics_value, identity_value]


def analysis_personality(args, test_data):
    all_data = []
    result_file = args.results_file
    cat = ['Personality Type', 'Role', 'Extraverted', 'Intuitive', 'Thinking', 'Judging', 'Assertive']
    df = pd.DataFrame(columns=cat)

    for case in test_data:
        ordered_list = [case[key]-4 for key in sorted(case.keys())]
        all_data.append(ordered_list)
        result = query_16personalities_api(ordered_list)
        result = result[:2] + tuple(result[2])
        df.loc[len(df)] = result
    
    column_sums = [sum(col) for col in zip(*all_data)]
    avg_data = [int(sum / len(all_data)) for sum in column_sums]
    avg_result = query_16personalities_api(avg_data)
    avg_result = avg_result[:2] + tuple(avg_result[2])
    df.loc["Avg"] = avg_result
    
    # Writing the results into a text file
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("# 16 Personality Results\n\n")
        f.write(df.to_markdown())


def analysis_results(questionnaire, args):
    significance_level = args.significance_level
    testing_file = args.testing_file
    result_file = args.results_file
    model = args.model
    
    test_data = convert_data(questionnaire, testing_file)
    
    if questionnaire["name"] == "16P":
        analysis_personality(args, test_data)
        return
    else:
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
    questionnaire_list = ['BFI', 'DTDD', 'EPQ-R', 'ECR-R', 'CABIN', 'GSE', 'LMS', 'BSRI', 'ICB', 'LOT-R', 'Empathy', 'EIS', 'WLEIS', '16P'] \
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

