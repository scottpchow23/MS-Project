import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import isnumeric

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

QUESTIONS_INDEX = {
  'Q24_1': 0,
  'Q24_2': 1,
  'Q24_3': 2,
  'Q24_4': 3,
  'Q24_5': 4,
  'Q25_1': 5,
  'Q26_1': 6,
  'Q27_1': 7,
  'Q28_1': 8,
  'Q29': 9,
  'Q30': 10,
}

QUESTION_TEXT_INDEX = {
  'Q24_1': 'I didn’t test my submission',
  'Q24_2': 'I used pre-written tests/feedback from Gradescope',
  'Q24_3': 'I added tests to the pre-written tests',
  'Q24_4': 'I manually tested the program by running it',
  'Q24_5': 'I wrote my own automated test suite',
  'Q25_1': 'I feel uncomfortable figuring out how to set up testing for projects and assignments.',
  'Q26_1': 'I feel that writing automated tests helps me catch more bugs in my code.',
  'Q27_1': 'I usually wait to get the project working before I write automated tests.',
  'Q28_1': 'I often do not write my own automated tests because I\'m unsure about how to set up tests/testing.',
  'Q29': 'Suppose you spend 10 hours on a programming assignment. How many hours do you think you would spend on setting up/writing automated tests?',
  'Q30': 'Suppose you spend 100 hours on a project. How many hours do you think you would spend on setting up/writing automated tests?',
}

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
        data.append([convert_raw_value(item) for item in response])

  column_indicies = []
  for question_id, index in column_id_index.items():
    if question_id in QUESTIONS:
      column_indicies.append(index)

  column_indicies.append(6) # add rfid column

  all_data = np.array([np.array(response) for response in data])
  return all_data[:, column_indicies], column_id_index, question_id_index

def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def convert_raw_value(raw_value):
  if raw_value in AGREEMENT_SCALE:
    return AGREEMENT_SCALE[raw_value]
  if raw_value in TIME_SCALE:
    return TIME_SCALE[raw_value]
  if is_float(raw_value):
    return float(raw_value)
  return raw_value

def compute(data, index, question_id_index):
  averages = []
  for question_id in QUESTIONS:
    question_index = index[question_id]
    average = data[:, question_index].astype(np.float).mean()
    averages.append(average)
    print(f'{question_id} "{question_id_index[question_id]}"" average: {average:.4f}')
  return averages


def compare(pre_data, pre_index, post_data, post_index):
  total_deltas = [0.0] * len(QUESTIONS)
  matched_response_count = 0
  seen = {}
  all_deltas = []
  for pre_response_index, pre_response in enumerate(pre_data):
    id = pre_response[-1]
    for post_response_index, post_response in enumerate(post_data):
      if id == post_response[-1] and id not in seen:
        seen[id] = True
        matched_response_count += 1
        response_deltas = []
        for question_index, question_id in enumerate(QUESTIONS):
          delta = convert_raw_value(post_response[post_index[question_id]]) - convert_raw_value(pre_response[pre_index[question_id]])
          response_deltas.append(delta)
          total_deltas[question_index] += delta
        all_deltas.append(response_deltas)

  np_all_deltas = np.array(all_deltas)

  print(f'{matched_response_count}/{min(len(pre_data), len(post_data))} matches responses in pre and post datasets.')
  return np_all_deltas

def plot(data, index, question_text_index, plot_folder):
  question_indicies = [index[question] for question in QUESTIONS]

  for question_id, question_index in zip(QUESTIONS, question_indicies):
    raw_question_data = data[:, question_index] # get the column of data associated with a question
    question_data = np.array([convert_raw_value(response) for response in raw_question_data]) # convert raw values to numbers
    question_text = question_text_index[question_id]

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
  print('pre-test averages')
  pretest_avgs = compute(pretest_data, QUESTIONS_INDEX, QUESTION_TEXT_INDEX)
  print('post-test averages')
  pretest_avgs = compute(posttest_data, QUESTIONS_INDEX, QUESTION_TEXT_INDEX)
  delta_data = compare(pretest_data, QUESTIONS_INDEX, posttest_data, QUESTIONS_INDEX)

  plot(pretest_data, QUESTIONS_INDEX, QUESTION_TEXT_INDEX, 'plots/pre')
  plot(posttest_data, QUESTIONS_INDEX, QUESTION_TEXT_INDEX, 'plots/post')
  plot(delta_data, QUESTIONS_INDEX, QUESTION_TEXT_INDEX, 'plots/delta')


if __name__ == '__main__':
  run()
