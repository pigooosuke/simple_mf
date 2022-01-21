import random
import pathlib


def main():
    dataset = list()
    train_dataset = list()
    test_dataset = list()
    with open("ml-1m/ml-1m/ratings.dat", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # user_id, item_id, rating, timestamp
                dataset.append(list(map(int,line.split("::"))))
    dataset = sorted(dataset, key=lambda x: x[3], reverse=True)
    checked_users = set()
    checked_items = set()
    train_items = dict()  # test item_id
    test_items = dict()  # test item_id
    for row in dataset:
        user = row[0]
        item = row[1]
        checked_items.add(item)
        if user not in checked_users:
            test_dataset.append(row)
            checked_users.add(user)
            test_items[user] = item
        else:
            train_dataset.append(row)
            train_items.setdefault(user, []).append(item)
    print(f"{len(train_dataset)=}")
    print(f"{len(test_dataset)=}")
    checked_items = list(checked_items)
    test_negatives = dict()
    for user in checked_users:
        user_train_items = train_items[user]
        user_test_items = test_items[user]
        user_test_negatives = random.sample([i for i in checked_items if i != user_test_items and i not in user_train_items], 100)
        test_negatives[user] = user_test_negatives

    pathlib.Path('data').mkdir(exist_ok=True)
    with open("data/ml-1m.train.rating", "w") as f:
        for row in train_dataset:
            user_id, item_id, rating, timestamp = row
            f.write(f"{user_id}\t{item_id}\t{rating}\t{timestamp}\n")
    with open("data/ml-1m.test.rating", "w") as f:
        for row in test_dataset:
            user_id, item_id, rating, timestamp = row
            f.write(f"{user_id}\t{item_id}\t{rating}\t{timestamp}\n")
    with open("data/ml-1m.test.negative", "w") as f:
        for user, items in test_negatives.items():
            user_test_items = test_items[user]
            negative_items = '\t'.join([str(i)for i in items])
            f.write(f"({user_id},{user_test_items})\t{negative_items}\n")


if __name__ == "__main__":
    main()