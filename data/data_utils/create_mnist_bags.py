import argparse
import numpy as np
import os
import torch


def main():
    args = get_args()
    r = np.random.RandomState(args.seed)

    mnist_labels = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_labels.npy"))

    #ranges from where to pick the training/test samples
    range_train = (0, 60000)
    range_test = (60000, 70000)

    label_of_last_bag = 0

    # create and store training bags
    bags_train_created = 0
    bag_ids_train = []
    while bags_train_created < args.num_bag_train:
        bag_length = np.int(r.normal(args.mean_bag_length, args.var_bag_length, 1))
        if bag_length < 1:
            bag_length = 1
        
        indices = r.randint(range_train[0], range_train[1], bag_length)
        labels_in_bag = mnist_labels[indices]

        if (args.target_number in labels_in_bag) and (label_of_last_bag == 0):
            # only create a positive bag if the previous one has been negative
            bag_ids_train.append(indices)
            label_of_last_bag = 1
            bags_train_created += 1
        elif label_of_last_bag == 1:
            # force the creation of a negative bag, if the last one has been positive
            index_list = []
            bag_length_counter = 0
            while bag_length_counter < bag_length:
                index = r.randint(range_train[0], range_train[1], 1)
                label_temp = mnist_labels[index]
                if label_temp != args.target_number:
                    index_list.append(index)
                    bag_length_counter += 1

            index_list = np.array(index_list)
            bag_ids_train.append(index_list)
            label_of_last_bag = 0
            bags_train_created += 1
        else:
            pass
    
    # TODO: storing of bag_ids
    np.savez(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_bag_ids_train.npz"), **{f"{i}":l for (i,l) in enumerate(bag_ids_train)})

    # create and store test bags
    bags_test_created = 0
    bag_ids_test = []
    while bags_test_created < args.num_bag_test:
        bag_length = np.int(r.normal(args.mean_bag_length, args.var_bag_length, 1))
        if bag_length < 1:
            bag_length = 1
        
        indices = r.randint(range_test[0], range_test[1], bag_length)
        labels_in_bag = mnist_labels[indices]

        if (args.target_number in labels_in_bag) and (label_of_last_bag == 0):
            # only create a positive bag if the previous one has been negative
            bag_ids_test.append(indices)
            label_of_last_bag = 1
            bags_train_created += 1
        elif label_of_last_bag == 1:
            # force the creation of a negative bag, if the last one has been positive
            index_list = []
            bag_length_counter = 0
            while bag_length_counter < bag_length:
                index = r.randint(range_test[0], range_test[1], 1)
                label_temp = mnist_labels[index]
                if label_temp != args.target_number:
                    index_list.append(index)
                    bag_length_counter += 1

            index_list = np.array(index_list)
            bag_ids_test.append(index_list)
            label_of_last_bag = 0
            bags_test_created += 1
        else:
            pass

    # TODO: storing of bag_ids
    np.savez(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_bag_ids_test.npz"), **{f"{i}":l for (i,l) in enumerate(bag_ids_test)})


    

def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-number', default=9, type=int)
    parser.add_argument('--mean-bag-length', default=10, type=int)
    parser.add_argument('--var-bag-length', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num-bag-train', default=1000, type=int)
    parser.add_argument('--num-bag-test', default=1000, type=int)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()