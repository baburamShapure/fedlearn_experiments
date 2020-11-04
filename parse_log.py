"""
parse log files 
"""

import os 
import pandas as pd 


contents = []
for file in os.listdir('logs'): 
    fp = os.path.join('logs', file)
    with open(fp, 'r') as f : 
        tmp= f.read()
        print(tmp)
        contents.append(tmp)



performances = tmp.split('\n')[-13:]

epoch = performances[0].split('|')[0].split(':')[-1].strip()[:-1]


performances[0].split('|')[1].split(',')

out = []
for each_line in performances[:-1]:
    epoch = each_line.split('|')[0].split(':')[-1].strip()[:-1]
    for each_agent in    each_line.split('|')[1: ]:
        agent = each_agent.split(',')[0]
        fold = 0
        for each_perf in each_agent.split(',')[1: ]: 
            out.append(pd.DataFrame([[epoch, agent,  fold,  each_perf]], 
                        columns = ['epoch', 'agent', 'fold', 'accuracy']))
            fold+= 1

OUT = pd.concat(out)
OUT.to_clipboard()