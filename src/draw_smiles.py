from rdkit.Chem import Draw
from rdkit import Chem

input_file0 = './out/out0-1.txt'
input_file1 = './out/out1-1.txt'
output_dir = './out/images/1/'
output_0 = './out/newout0.txt'
output_1 = './out/newout1.txt'
num_collapse0, num_collapse1 = 0, 0


with open(input_file0, 'r') as f0:
    text0 = f0.read().split()
with open(input_file1, 'r') as f1:
    text1 = f1.read().split()

ans0, ans1 = '', ''
for i, seq in enumerate(text0):
    m = Chem.MolFromSmiles(seq)
    print(seq, '>>>', m)
    if m == None:
        num_collapse0 += 1
        continue
    ans0 += seq + '\n'
    Draw.MolToFile(m, output_dir+'/mol0_'+str(i)+'.png')
ans0 = ans0.rstrip()

for i, seq in enumerate(text1):
    m = Chem.MolFromSmiles(seq)
    if m == None:
        num_collapse1 += 1
        continue
    ans1 += seq + '\n'
    Draw.MolToFile(m, output_dir+'/mol1_'+str(i)+'.png')
ans1 = ans1.rstrip()

print('生成可能な数(toxicity=0)：　', str(10000-num_collapse0), ' / 10000')
print('生成可能な数(toxicity=1)：　', str(10000-num_collapse1), ' / 10000')


with open(output_0, 'w') as f:
    f.write(ans0)

with  open(output_1, 'w') as  f:
    f.write(ans1)
