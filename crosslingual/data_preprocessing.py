
from tqdm import tqdm
import json

lang_list = ["En", "Zh", "Fr", "Id", "It", "Jp", "Ko"]

for lang in lang_list:
    for split in ["train", "valid", "test"]:
        if lang == "En":
            file_name = "%s_persona_%s.json" % (lang, split)
        elif split == "train":
            file_name = "%s_persona_%s_corrected.json" % (lang, split)
        else:
            file_name = "%s_persona_split_%s_human_annotated.json" % (lang, split)
        print(file_name)
        with open("../dataset/" + file_name, 'rb') as json_file:
            data = json.load(json_file)
            file_out_x = open("data/xpersona/%s.x.%s" % (split, lang.lower()), "w")  # dialog history
            file_out_y = open("data/xpersona/%s.y.%s" % (split, lang.lower()), "w")  # response
            for each_dialog in tqdm(data):
                # preprocess each dialogue
                persona_list = each_dialog["persona"]
                persona_str = ""
                # persona
                for persona in persona_list:
                    persona_str = persona_str + persona + " "

                # dialogue
                dialogue_tuples = each_dialog["dialogue"]
                turns = []
                for tuple_ in dialogue_tuples:
                    for turn in tuple_:
                        turns.append(turn)

                for idx in range(len(turns)):
                    if idx % 2 == 0:
                        continue
                    if idx == 1:
                        user_turn = turns[idx-1]
                        system_turn = turns[idx]
                        file_out_x.write(persona_str + user_turn + "\n")
                        file_out_y.write(system_turn + "\n")
                    else:
                        user_turn1 = turns[idx-3]
                        system_turn1 = turns[idx-2]
                        user_turn2 = turns[idx-1]
                        system_turn2 = turns[idx]
                        file_out_x.write(persona_str + user_turn1 + " " + system_turn1 + " " + user_turn2 + "\n")
                        file_out_y.write(system_turn2 + "\n")

            file_out_x.close()
            file_out_y.close()
