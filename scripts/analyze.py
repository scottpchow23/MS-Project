import matplotlib.pyplot as plt
import numpy as np

AGREEMENT_SCALE = {
  'Strongly disagree': -2,
  'Somewhat disagree': -1,
  'Neither agree nor disagree': 0,
  'Somewhat agree': 1,
  'Strongly agree': 2
}

TIME_SCALE = {
  'Never': 0,
  'Sometimes': 1,
  'About half the time': 2,
  'Most of the time': 3,
  'Always': 4,
}

QUESTIONS = [
  'Q24_1',
  'Q24_2',
  'Q24_3',
  'Q24_4',
  'Q24_5',
  'Q25_1',
  'Q26_1',
  'Q27_1',
  'Q28_1',
  'Q29',
  'Q30',
]

def load_data_and_index(file_name):
  data = []
  column_id_index = {}
  question_id_index = {}
  with open(f'data/{file_name}') as f:
    for num, line in enumerate(f, 1):
      if num == 1:
        print(line)
        for index, id in enumerate(line.split('\t')):
          column_id_index[id] = index
      elif num == 2:
        print(line)
        for index, question_text in enumerate(line.split('\t')):
          question_id_index[index] = question_text
      elif num > 2:
        response = line.split('\t')
        if len(response) != 79:
          response.append('')
        data.append(response)

  return np.array([np.array(response) for response in data]), column_id_index, question_id_index

def convert_raw_value(raw_value):
  if raw_value in AGREEMENT_SCALE:
    return AGREEMENT_SCALE[raw_value]
  if raw_value in TIME_SCALE:
    return TIME_SCALE[raw_value]
  return float(raw_value)

def compute(data, index, question_id_index):
  averages = []
  for question_id in QUESTIONS:
    question_index = index[question_id]
    total = 0.0
    for response_index, response in enumerate(data):
      raw_value = response[question_index]
      if raw_value == '':
        raw_value = '0'
      value = convert_raw_value(raw_value)
      total += value
    average = total / len(data)
    averages.append(average)
    print(f'{question_id} "{question_id_index[question_index]}"" average: {average}')

  return averages


def compare(pre_data, pre_index, post_data, post_index):
  total_deltas = [0.0] * len(QUESTIONS)
  matched_response_count = 0
  seen = {}
  for pre_response_index, pre_response in enumerate(pre_data):
    id = pre_response[6]
    for post_response_index, post_response in enumerate(post_data):
      if id == post_response[6] and id not in seen:
        seen[id] = True
        matched_response_count += 1
        for question_index, question_id in enumerate(QUESTIONS):
          delta = convert_raw_value(post_response[post_index[question_id]]) - convert_raw_value(pre_response[pre_index[question_id]])
          total_deltas[question_index] += delta

  for pre_response_index, pre_response in enumerate(post_data):
    if pre_response[6] not in seen:
      print(pre_response_index, pre_response[6])

  if matched_response_count == 0:
    print('No matches found; trivially returning 0s for averages.')
    return total_deltas
  print(f'{len(pre_data)}')
  print(f'{matched_response_count} matches responses in pre and post datasets.')
  return [total / matched_response_count for total in total_deltas]

def plot(data, index, question_text_index, plot_folder):
  question_indicies = [index[question] for question in QUESTIONS]

  for question_id, question_index in zip(QUESTIONS, question_indicies):
    raw_question_data = data[:, question_index] # get the column of data associated with a question
    question_data = np.array([convert_raw_value(response) for response in raw_question_data]) # convert raw values to numbers
    question_text = question_text_index[question_index]
    # plt.subplot(x=question_data, bins='auto', label=question_text)
    fig, axes = plt.subplots()
    bins = np.arange(question_data.min(), question_data.max() + 1.5) - 0.5
    axes.hist(x=question_data, bins=bins)
    axes.title.set_text(question_text)
    axes.axvline(question_data.mean(), color='k', linestyle='dashed', linewidth=1)
    fig.savefig(f'{plot_folder}/{question_id}', bbox_inches='tight')
    plt.close(fig)

def run():
  pretest_data, pretest_index, pretest_question_text_index = load_data_and_index('pretest.tsv')
  posttest_data, posttest_index, posttest_question_text_index = load_data_and_index('posttest.tsv')
  pretest_averages = compute(pretest_data, pretest_index, pretest_question_text_index)
  posttest_averages = compute(posttest_data, posttest_index, posttest_question_text_index)
  deltas = compare(pretest_data, pretest_index, posttest_data, posttest_index)
  print(list(zip(QUESTIONS, deltas)))
  plot(pretest_data, pretest_index, pretest_question_text_index, 'plots/pre')
  plot(posttest_data, posttest_index, posttest_question_text_index, 'plots/post')

if __name__ == '__main__':
  run()
