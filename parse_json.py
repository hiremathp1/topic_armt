import json


def get_text(obj):
    obj = obj["Items"]
    # Sort text by start_time
    sorted_obj = sorted(obj, key=lambda x: x["start_time"])
    return " ".join([item["text"] for item in obj])


def test():
    return get_text(json.load(open("number_system.json")))


if __name__ == "__main__":
    text = test()
    print(text)
    print(text.split().__len__())
