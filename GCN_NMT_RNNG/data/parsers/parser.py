import re

class Node(object):
    def __init__(self, num, head, label):
        self.num = num
        self.head = head
        self.label = label


def write_oracle(buffer, buffer_child_to_head_dict):
    arcs = []
    stack = []
    while len(buffer) > 0 or len(stack) != 1:
        if len(stack) > 1 and stack[-1].num == stack[-2].head:
            arcs.append('REDUCE-LEFT-ARC(' + stack[-1].label + ')')
            stack.pop(-2)
        elif len(stack) > 1 and stack[-2].num == stack[-1].head:
            if stack[-1].num in buffer_child_to_head_dict.values():
                arcs.append('SHIFT')
                del buffer_child_to_head_dict[buffer[0].num]
                stack.append(buffer.pop(0))
            else:
                arcs.append('REDUCE-RIGHT-ARC(' + stack[-2].label + ')')
                stack.pop(-1)
        elif len(buffer) > 0:
            arcs.append('SHIFT')
            del buffer_child_to_head_dict[buffer[0].num]
            stack.append(buffer.pop(0))
        else:
            # stack에 독립 item 존재
            break
    return arcs


all_oracles = open('../train.oracle.en', 'w', encoding='utf-8')

tagged_filename = '../tagged_train.en'
tagged_file = open(tagged_filename, 'r', encoding='utf-8')
bulk = tagged_file.read()
blocks = re.compile(r"\n{2,}").split(bulk)
blocks = list(filter(None, blocks))
sentences = []
cnt = 0
none = 0
for block in blocks:
    tokens = []
    buffer = []
    child_to_head_dict = {}
    for line in block.splitlines():
        attr_list = line.split('\t')
        tokens.append(attr_list[1])
        num = int(attr_list[0])
        head = int(attr_list[6])
        label = attr_list[7]
        node = Node(num, head, label)
        child_to_head_dict[num] = head
        buffer.append(node)
    arcs = write_oracle(buffer, child_to_head_dict)
    print(tokens)
    for arc in arcs:
        print(arc)
        all_oracles.write(arc + '\n')
    print('\n')
    all_oracles.write('\n')
    cnt += 1
    print(cnt)
tagged_file.close()
all_oracles.close()
print('none?')
print(none)



















