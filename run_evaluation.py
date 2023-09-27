from utils.get_accuracy import *

# raw data path (Not the preprocessed data path) 
raw_data_path_ = ''
# the directory where we put the excel files: accuracy.xlsx, sub_task_label.xslx
# usually found in the prediction log path: logs/{experiment name}/version_{version_number}
output_root_path_ = ''

assert raw_data_path_, f'variable <raw_data_path_> variable cannot be empty'
assert output_root_path_, f'variable <output_root_path_> variable cannot be empty'

# excel file names
subtask_label_path_ = output_root_path_ + '/sub_task_label.xlsx'
accuracy_file_path_ = output_root_path_ + '/accuracy.xlsx'
save_sub_task_acc_file_path_ = output_root_path_ + '/sub_task_accuracy.xlsx'

if __name__ == '__main__':
    # Calculate the task accuracy
    hierarchy = load_data_hierarachy(raw_data_path_)
    data = load_excel(subtask_label_path_)

    # Obtain task accuracy and subtask accuracy
    all_task_acc, sub_task_acc = get_task_acc(data, hierarchy)

    # Write statistics to excel file
    append_excel(accuracy_file_path_, all_task_acc)
    save_excel(save_sub_task_acc_file_path_, sub_task_acc)
