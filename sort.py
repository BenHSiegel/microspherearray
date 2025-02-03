import numpy as np
import time
def timed_sort_by_second_array(arr1, arr2):
    start_time = time.time()
    result = sort_by_second_array(arr1, arr2)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return result
def sort_by_second_array(arr1, arr2):
    # Create a dictionary to hold lists for each unique value in arr2
    sorted_dict = {}
    for value in set(arr2):
        sorted_dict[value] = []

    # Combine the two arrays into a list of tuples
    combined = list(zip(arr1, arr2))

    # Sort the combined list by the second element of the tuples
    sorted_combined = sorted(combined, key=lambda x: x[1])

    # Distribute the sorted elements into the dictionary
    for item in sorted_combined:
        sorted_dict[item[1]].append(item[0])

    return sorted_dict


# Example usage
arr1 = [i for i in range(5000)]
arr2 = [i % 10 for i in range(5000)]
np.random.shuffle(arr2)
sorted_arr1 = timed_sort_by_second_array(arr1, arr2)
print(sorted_arr1)
