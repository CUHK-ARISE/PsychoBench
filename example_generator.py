import openai
import os
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time
from tqdm import tqdm


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat(
    model,           # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
    messages,        # [{"role": "system"/"user"/"assistant", "content": "Hello!", "name": "example"}]
    temperature=0,   # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    n=1,             # Chat completion choices to generate for each input message.
    max_tokens=1024, # The maximum number of tokens to generate in the chat completion.
    delay=5          # Seconds to sleep after each request.
):
    time.sleep(delay)
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    
    if n == 1:
        return response['choices'][0]['message']['content']
    else:
        return [i['message']['content'] for i in response['choices']]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion(
    model,           # text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
    prompt,          # The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
    temperature=0,   # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    n=1,             # Completions to generate for each prompt.
    max_tokens=1024, # The maximum number of tokens to generate in the chat completion.
    delay=10         # Seconds to sleep after each request.
):
    time.sleep(delay)
    
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    
    if n == 1:
        return response['choices'][0]['text']
    else:
        response = response['choices']
        response.sort(key=lambda x: x['index'])
        return [i['text'] for i in response['choices']]


def convert_results(result, column_header):
    result = result.strip()  # Remove leading and trailing whitespace
    try:
        result_list = [int(element.strip()[-1]) for element in result.split('\n') if element.strip()]
    except:
        result_list = ["" for element in result.split('\n')]
        print(f"Unable to capture the responses on {column_header}.")
        
    return result_list


def example_generator(questionnaire, args):
    testing_file = args.testing_file
    model = args.model
    records_file = args.name_exp if args.name_exp is not None else model

    openai.organization = args.openai_organization
    openai.api_key = args.openai_key

    # Read the existing CSV file into a pandas DataFrame
    df = pd.read_csv(testing_file)

    # Find the columns whose headers start with "order"
    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    with tqdm(total=total_iterations) as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                # Find the index of the previous column
                questions_column_index = i - 1
                shuffle_count += 1
                
                # Retrieve the column data as a string
                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [questions_list[i:i+40] for i in range(0, len(questions_list), 40)]  
                questions_list = ['\n'.join(questions) for questions in separated_questions]


                for k in range(args.test_count):
                    
                    df = pd.read_csv(testing_file)
                    
                    # Insert the updated column into the DataFrame with a unique identifier in the header
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'
                    
                    result_string_list = []
                    previous_records = []
                    
                    for questions_string in questions_list:
                        result = ''
                        if model == 'text-davinci-003':
                            inputs = questionnaire["prompt"] + '\n' + questions_string
                            result = completion(model, inputs)
                        elif model in ['gpt-3.5-turbo', 'gpt-4']:
                            inputs = previous_records + [
                                {"role": "system", "content": questionnaire["inner_setting"]},
                                {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                            ]
                            result = chat(model, inputs)
                            previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                            previous_records.append({"role": "assistant", "content": result})
                        else:
                            raise ValueError("The model is not supported or does not exist.")
                    
                        result_string_list.append(result.strip())
                    
                        # Write the prompts and results to the file
                        os.makedirs("prompts", exist_ok=True)
                        os.makedirs("responses", exist_ok=True)

                        with open(f'prompts/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                            file.write(f'{inputs}\n====\n')
                        with open(f'responses/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                            file.write(f'{result}\n====\n')
                        
                        
                    result_string = '\n'.join(result_string_list)
                    
                    result_list = convert_results(result_string, column_header)
                    
                    try:
                        if column_header in df.columns:
                            df[column_header] = result_list
                        else:
                            df.insert(i + insert_count + 1, column_header, result_list)
                            insert_count += 1
                    except:
                        print(f"Unable to capture the responses on {column_header}.")

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(testing_file, index=False)
                    
                    pbar.update(1)
