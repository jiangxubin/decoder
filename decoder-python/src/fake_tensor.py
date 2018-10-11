import numpy as np
import os


def fake_tensor(file_list: list, candidate_size=500):
    """
    Now that I have no batch tensor,write this function to merge several nps into a tensor manually for use of multi threads decoder test.
    :param batch_size: number of training examples in a batch which equals to the first element of tensor's shape.
    :param file_list: list of file paths which will be used to merge.
    :param candidate_size: Number of highest probability candidates to be filtered out of all possibilities.
    :return: list which stores the time-step of each example in a batch
    """
    max_time_step = 0
    index_set = list()
    flattened_index_set = list()
    # index_txt = open(r"/home/augustus/Documents/decoder/decoder-cpp/data/multi_thread_test_data/index.txt", 'w')
    # value_txt = open(r"/home/augustus/Documents/decoder/decoder-cpp/data/multi_thread_test_data/value.txt", 'w')
    value_set = list()
    flattend_value_set = list()
    time_steps = list()
    for path in file_list:
        data = np.load(path)
        time_step = data.shape[0]
        time_steps.append(time_step)
        if time_step > max_time_step:
            max_time_step = time_step
        index = np.zeros((data.shape[0], candidate_size))
        value = np.zeros((data.shape[0], candidate_size))
        for i in range(data.shape[0]):
            partition_index = np.argpartition(-data[i, :], candidate_size)
            top_index = partition_index[:candidate_size]
            index[i, :] = top_index
            partition_value = np.partition(-data[i, :], candidate_size)
            top_value = -partition_value[:candidate_size]
            value[i, :] = top_value
        index_set.append(index)
        value_set.append(value)
    for index in index_set:
        remaining = np.zeros((max_time_step-index.shape[0], index.shape[1]))
        index = np.concatenate((index, remaining), axis=0)
        flattened_index = index.reshape((1, -1))
        flattened_index_set.append(flattened_index)

    for value in value_set:
        remaining = np.zeros((max_time_step-value.shape[0], value.shape[1]))
        value = np.concatenate((value, remaining), axis=0)
        flattened_value = value.reshape((1, -1))
        flattend_value_set.append(flattened_value)

    single_index = np.concatenate(flattened_index_set, axis=1)
    single_value = np.concatenate(flattend_value_set, axis=1)
    np.savetxt(r"/home/augustus/Documents/decoder/decoder-cpp/data/multi_thread_test_data/index.txt", single_index, delimiter=' ')
    np.savetxt(r"/home/augustus/Documents/decoder/decoder-cpp/data/multi_thread_test_data/value.txt", single_value, delimiter=' ')
    time_steps = np.array(time_steps).reshape((1, -1))
    np.savetxt(r"/home/augustus/Documents/decoder/decoder-cpp/data/multi_thread_test_data/length.txt", time_steps, delimiter=' ')



if __name__ == "__main__":
    test_list = [r"/home/augustus/Documents/decoder/decoder-python/data/nps/38.npy", r"/home/augustus/Documents/decoder/decoder-python/data/nps/40.npy",
                 r"/home/augustus/Documents/decoder/decoder-python/data/nps/49.npy", r"/home/augustus/Documents/decoder/decoder-python/data/nps/59.npy"
                 ]
    fake_tensor(test_list)



