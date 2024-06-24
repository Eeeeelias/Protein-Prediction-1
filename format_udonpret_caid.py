

outputs = {}
with open('data/udonpred_original_all.caid', 'r') as f:
    current_chain = []
    current_id = ''
    for line in f.readlines():
        if line.startswith('>'):
            if current_chain:
                outputs[current_id] = current_chain
            current_id = line.strip()[1:]
            current_chain = []
        else:
            current_chain.append(float(line.strip().split("\t")[2]))
    outputs[current_id] = current_chain

with open('data/udonpred_original_all.tsv', 'w') as f:
    for key, value in outputs.items():
        f.write(f"{key}\t{value}\n")
