import re
from rouge import Rouge
import argparse
import os
import json
import numpy as np
import sys
sys.setrecursionlimit(10000)

class Eval:

    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def char(self, index):
        if index < 26:
            return chr(index+65)
        elif index < 52:
            return 'A'+chr(index+65-26)
        else:
            return 'B'+chr(index+65-26-26)

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip().lower()
        return answer
    
    @classmethod
    def evaluate_rouge(cls, predictions):
        scorer = cls()
        rouge = Rouge()
        acc = {'f': []}
        eval_list = []

        with open(predictions, 'r') as f:
            for line in f:
                sample = json.loads(line)
                raw_ans = sample['answer']
                raw_pred = sample['pred']
                gt_ans = scorer.process(sample['answer'])
                pred_ans = scorer.process(sample['pred'])
                str_match = int(gt_ans == pred_ans)
                # print(f'gt_ans: {gt_ans}, pred_ans: {pred_ans}')
                if pred_ans == '':
                    score = 0
                else:
                    p = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['p']
                    r = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['r']
                    beta = 1
                    if p == 0 and r == 0:
                        score = 0
                    else:
                        score = ((1 + beta**2) * p * r) / (beta**2 * p + r)

                acc['f'].append(score)
                eval_list.append({
                    'id': str(sample['id']),
                    'ans': raw_ans,
                    'pred': raw_pred,
                    'Rouge-L': str(round(score, 3)),
                    'str_match': str_match,
                })

        return {
            'Rouge-L f': np.round(np.mean(acc['f']), 6),
            'eval_list': eval_list
        }

    # 字符串匹配 返回0或者1（不区分大小写）
    @classmethod
    def evaluate_str_match(cls, predictions):
        scorer = cls()
        acc = []
        with open(predictions, 'r') as f:
            for line in f:
                sample = json.loads(line)
                gt_ans = scorer.process(sample['answer'])
                pred_ans = scorer.process(sample['pred'])
                acc.append(int(gt_ans == pred_ans))
        return {
            'str_match': np.round(np.mean(acc), 6)
        }

def main(args):
    # pass
    scorer = Eval()
    rouge_result = scorer.evaluate_rouge(args.pred_path)
    str_match_result = scorer.evaluate_str_match(args.pred_path)
    eval_result = {
        'Rouge-L f': rouge_result['Rouge-L f'],
        'str_match': str_match_result['str_match'],
        'eval_list': rouge_result['eval_list'],
    }

    print(f"Rouge-L: {eval_result['Rouge-L f']}")
    print(f"String Match: {eval_result['str_match']}")
    with open(args.log_path, 'a') as f:
        f.write(f"Rouge-L: {eval_result['Rouge-L f']}")
        f.write(f"String Match: {eval_result['str_match']}")
    json.dump(eval_result, open(os.path.join(args.output_dir, 'eval_result.json'),'w'), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--log-path', type=str, required=True)
    args = parser.parse_args()

    main(args)