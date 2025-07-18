"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tqdm import tqdm
import json
import pandas as pd
import transformers
from huggingface_hub import login
import os
from collections import defaultdict
import torch

intro_prompt = "Here is a list of items a user bought along with their respective reviews in json format:\n {}\n"
new_system_prompt = "\nYour task is to generate a list of up to five search instructions that reflect the user's preferences based on their reviews. Be specific on what the user likes, does not like, and should be avoided. Do not mention brands or certain products. Return a json file containing the search instructions with key 'instructions'. Keep the instructions simple, short and concise, and do NOT include comments on delivery time or pricing."
prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{}"
query_prompt = "<|eot_id|><|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\nHere is the generated list of search instructions in JSON format:"

datasets = ["Beauty", "Sports_and_Outdoors", "Toys_and_Games", "yelp", "steam"]
for dataset in datasets:
    metadata = json.load(open(f"item2review_{dataset}.json"))
    metadata = {userid: metadata[userid][-20:] for userid in metadata.keys()}
    if dataset == "steam":
        user_ids = list(metadata.keys())
        user_ids = user_ids[::7]
        metadata = {uid: metadata[uid] for uid in user_ids}
    instruction_list = []
    for user_id in tqdm(metadata.keys()):
        item_list = []
        for item in metadata[user_id]:
            item_dict = item.copy()
            del item_dict["itemid"]
            del item_dict["userid"]
            if dataset == "Toys_and_Games":
                del item_dict["title"]
                del item_dict["price"]
            item_list.append("```{}```".format(json.dumps(item_dict)))
            if len(item_list) >= 5:
                item_list = item_list[-5:]
            cur_prompt = (
                prompt.format(
                    intro_prompt.format("\n".join(item_list)) + new_system_prompt
                )
                + query_prompt
            )
            instruction_list.append((user_id, item, cur_prompt, new_system_prompt))
    instruction_dict = {
        "user_id": [item[0] for item in instruction_list],
        "item_id": [item[1] for item in instruction_list],
        "prompt": [item[2] for item in instruction_list],
    }
    pref_df = pd.DataFrame(data=instruction_dict)
    id2user_and_item_id = {
        i: (user_id, item_id)
        for i, (user_id, item_id) in enumerate(
            zip(pref_df["user_id"].values, pref_df["item_id"].values)
        )
    }
    pref_df = pref_df.drop(["user_id", "item_id"], axis=1)
    pref_df["uid"] = pref_df.index
    pref_df.to_csv(f"pref_df_history_to_instruct_{dataset}.csv")

    with open(f"id2user_and_item_id_{dataset}.json", "w") as f:
        json.dump(id2user_and_item_id, f)

if not os.path.exists(os.path.expanduser('~/.cache/huggingface/token')):
    print("llama3 models require authorized access... exiting")
    exit(1)
# login with hf authentication token
auth_token = open(os.path.expanduser('~/.cache/huggingface/token'), 'r').read()
login(auth_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
gen_kwargs = {'top_p': 0.9, "temperature": 0.6, "max_new_tokens": 512, "repetition_penalty": 1}
pipeline = transformers.pipeline("text-generation", model="meta-llama/Meta-Llama-3-70B-Instruct",
                                 model_kwargs={'cache_dir': 'system/user/publicdata/llm'},
                                 device_map="auto",
                                 **gen_kwargs)

for dataset in datasets:
    result_dict = defaultdict(dict)
    responses = []
    # load dataframe
    pref_df = pd.read_csv(f"pref_df_history_to_instruct_{dataset}.csv")
    pref_arr = pref_df.values
    uid_to_useritemid = json.load(open(f"id2user_and_item_id_{dataset}.json", "r"))    

    pref_arr = pref_arr[:2]
    # prepare input batches and feed to model
    for i in range(len(pref_arr)):
        uid = str(pref_arr[i][0])
        prompt = pref_arr[i][1]
        res = pipeline(prompt)[0]['generated_text'][len(prompt):]
        responses.append((uid, res))

    failed_prompts = []
    # needs adjustment to actual pandas dataframe!
    for row in responses:
        try:
            res = json.loads(row[1].split("```")[1].replace("\\n", "").replace("\\", ""))
            orig_dict = uid_to_useritemid[row[0]]
            assert isinstance(res["instructions"], list) and isinstance(res["instructions"][0], str) and len(res["instructions"]) >= 5
            if len(res["instructions"]) > 5:
                res["instructions"] = res["instructions"][:5]
            assert not any([len(res["instructions"][k].split(' ')) == 1 for k in range(5)])
            result_dict[orig_dict[1]["userid"]][orig_dict[1]["itemid"]] = res[
                "instructions"
            ]
        except:
            failed_prompts.append(row)

    second_failed = []
    for row in failed_prompts:
        try:
            res = [
                l[1:]
                for l in row[1][row[1].index("{") : row[1].index("}") + 1].split("\\")
                if len(l) > 1
                and l[1].isalpha()
                and l != '"instructions'
                and not "user_id" in l
                and not "item_id" in l
            ]
            res = [l for l in res if l != "id" and l != "instruction" and l != "all"]
            assert len(res) >= 5
            if len(res) > 5:
                res = res[:5]
            assert not any([len(res[k].split(" ")) == 1 for k in range(5)])
            orig_dict = uid_to_useritemid[row[0]]
            result_dict[orig_dict[1]["userid"]][orig_dict[1]["itemid"]] = res
        except KeyError:
            print(f"id mismatch!")
        except:
            second_failed.append(row)

    still_failed = []
    for row in second_failed:
        try:
            try:
                res = [r.replace("\\", '').strip() for r in row[1].split("instructions")[1].split("[")[1].split("]")[0].split('"')if len(r) and r[0].isalpha()]
            except:
                res = [r for r in still_failed[0][1].split("instrucciones")[1].split('[')[1].split(']')[0].split('"') if len(r) and r[0].isalpha()]
            orig_dict = uid_to_useritemid[row[0]]
            assert isinstance(res, list)
            res = [l for l in res if l != 'id' and l != 'instruction' and l != 'all']
            assert len(res) >= 5
            if len(res) > 5:
                res = res[:5]
            assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
            result_dict[orig_dict[1]["userid"]][orig_dict[1]["itemid"]] = res
        except:
            still_failed.append(row)

    failed_uids = [still_failed[i][0] for i in range(len(still_failed))]
    response_list = [still_failed[i][1] for i in range(len(still_failed))]
    even_failed_here = []
    for i, (uid, resp) in enumerate(zip(failed_uids, response_list)):
        try:
            orig_dict = uid_to_useritemid[str(uid)][1]
            if resp == '':
                raise Exception
            try:
                res = json.loads(resp.split('```')[1].replace('\\n', '').replace('\\', ''))
                assert isinstance(res, dict) and len(res['instructions']) >= 5 and isinstance(res['instructions'][0], str)
                assert not any([len(res['instructions'][k].split(' ')) == 1 for k in range(5)])
                result_dict[orig_dict['userid']][orig_dict['itemid']] = res['instructions']
            except:
                try:
                    res = [r.replace("\\", '').strip() for r in resp.split('"instructions')[1].split("[")[1].split("]")[0].split('"') if len(r) and r[0].isalpha()]
                    assert isinstance(res, list) and len(res) >= 5 and isinstance(res[0], str)
                    assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
                    result_dict[orig_dict['userid']][orig_dict['itemid']] = res[:5]
                except:
                    try:
                        res = [r.replace("\\", '').strip() for r in resp.split('"instructions"')[1].split("[")[1].split("]")[0].split('"') if len(r) and r[0].isalpha()]
                        assert isinstance(res, list) and len(res) >= 5 and isinstance(res[0], str)
                        assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
                        result_dict[orig_dict['userid']][orig_dict['itemid']] = res[:5]
                    except:
                        try:
                            res = [r.replace("\\", '').strip() for r in resp.split('"instructions"')[1].split("[")[1].split("]")[0].split('"') if len(r) and r[0].isalpha()]
                            assert isinstance(res, list) and len(res) >= 5 and isinstance(res[0], str)
                            assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
                            result_dict[orig_dict['userid']][orig_dict['itemid']] = res[:5]
                        except:
                            try:
                                res = eval(resp[resp.index('[') : resp.index(']')+1])
                                assert isinstance(res, list) and len(res) >= 5
                                if isinstance(res[0], dict):
                                    res = [r['instruction'] for r in res]
                                assert isinstance(res[0], str)
                                assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
                                result_dict[orig_dict['userid']][orig_dict['itemid']] = res[:5]
                            except Exception as e:
                                print(repr(e))
                                print(i)
                                even_failed_here.append(i)
        except:
            even_failed_here.append(i)

    failed_uids = [failed_uids[i] for i in even_failed_here]
    response_list = [response_list[i] for i in even_failed_here]
    finally_failed = []
    for i, (uid, resp) in enumerate(zip(failed_uids, response_list)):
        try:
            orig_dict = uid_to_useritemid[str(uid)][1]
            if resp == '':
                raise Exception
            try:
                res = json.loads(resp.split('```')[1].replace('\\n', '').replace('\\', ''))
                assert isinstance(res, dict) and len(res['instructions']) >= 5 and isinstance(res['instructions'][0], str)
                assert not any([len(res['instructions'][k].split(' ')) == 1 for k in range(5)])
                result_dict[orig_dict['userid']][orig_dict['itemid']] = res['instructions']
            except:
                try:
                    res = [r.replace("\\", '').strip() for r in resp.split('"instructions')[1].split("[")[1].split("]")[0].split('"') if len(r) and r[0].isalpha()]
                    assert isinstance(res, list) and len(res) >= 5 and isinstance(res[0], str)
                    assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
                    result_dict[orig_dict['userid']][orig_dict['itemid']] = res[:5]
                except:
                    try:
                        res = [r.replace("\\", '').strip() for r in resp.split('"instructions"')[1].split("[")[1].split("]")[0].split('"') if len(r) and r[0].isalpha()]
                        assert isinstance(res, list) and len(res) >= 5 and isinstance(res[0], str)
                        assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
                        result_dict[orig_dict['userid']][orig_dict['itemid']] = res[:5]
                    except:
                        try:
                            res = [r.replace("\\", '').strip() for r in resp.split('"instructions"')[1].split("[")[1].split("]")[0].split('"') if len(r) and r[0].isalpha()]
                            assert isinstance(res, list) and len(res) >= 5 and isinstance(res[0], str)
                            assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
                            result_dict[orig_dict['userid']][orig_dict['itemid']] = res[:5]
                        except:
                            try:
                                res = eval(resp[resp.index('[') : resp.index(']')+1])
                                assert isinstance(res, list) and len(res) >= 5 
                                if isinstance(res[0], dict):
                                    res = [r['instruction'] for r in res]
                                assert isinstance(res[0], str)
                                assert not any([len(res[k].split(' ')) == 1 for k in range(5)])
                                result_dict[orig_dict['userid']][orig_dict['itemid']] = res[:5]
                            except Exception as e:
                                print(repr(e))
                                finally_failed.append(i)
        except:
            finally_failed.append(i)

    for user in result_dict.keys():
        for item in result_dict[user].keys():
            if any([len(result_dict[user][item][k].split(' ')) == 1 for k in range(5)]):
                print(f"User: {user}, Item: {item}, Inst: {result_dict[user][item]}")

    with open(f"preference_dict_{dataset}.json", "w") as f:
        json.dump(result_dict, f)
