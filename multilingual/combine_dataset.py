import json

LANGUAGE = ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]
dataset = {"train":{},"valid":{}, "test":{}} # {"train":{"En":[...], "Id":[...]},"valid":{}, "test":{}}
for split in dataset.keys():
    for lang in LANGUAGE:
        dataset[split][lang] = []
        if split=="train":
            path = "../dataset/{}_persona_{}_corrected.json".format(lang, split)
        else:
            path = "../dataset/{}_persona_split_{}_human_annotated.json".format(lang, split)
        if lang=="En":
            path = "../dataset/{}_persona_{}.json".format(lang, split)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for dial in data:  #{"persona":[], "dialogue":[[], [],]}
                dialogue_history = []
                for turn in dial["dialogue"]:
                    dialogue_history.append(turn[0])
                    dataset[split][lang].append({"persona":dial["persona"], "history":dialogue_history.copy(), "response":turn[1]})
                    dialogue_history.append(turn[1]) #add response
        print(dataset[split][lang][:30])

with open("multilingual_new.json", "w", encoding="utf-8") as f:
    json.dump(dataset,f)


# dataset = []
# with open("../multilingual_data/Zh_persona_train_corrected.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#     for dial in data:
#         dialogue_history = []
#         for turn in dial["dialogue"]:
#             dialogue_history.append(turn[0])
#             dataset.append({"history":dialogue_history.copy(), "response":turn[1]})
#             dialogue_history.append(turn[1]) #add response
#     print(dataset[:500])
# with open("zh_persona.json", "w", encoding="utf-8") as f:
#     json.dump(dataset,f)
