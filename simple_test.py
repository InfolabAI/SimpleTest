import torch
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace

SEED=1223

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def normal_sampling(mu1, v1, mu2, v2, nrow, label):
    #ret normal dist with mu1, v1 -> x0=[,,,,,,] num_element=nrow
    #x1=[,,,,,,], label=[,,,,,,]
    x0 = torch.normal(mu1, v1, size=(1, nrow)).view(-1)
    x1 = torch.normal(mu2, v2, size=(1, nrow)).view(-1)
    label = torch.LongTensor(nrow).fill_(label)
    return [x0, x1, label]

def draw_examples(x0, x1, label, file_name):
    #NOTE font size
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=30)              # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #x1=[,,,,,,], label=[,,,,,,]
    #NOTE get_cmap("jet", num): num means how many color you need to present examples. 3 means {0:target label, 1:target label, 2:distractor}
    if 2 not in label.unique():
        plt.scatter(x0, x1, c=label, s=500, cmap=plt.cm.get_cmap("jet",3), marker='.')
    if 2 in label.unique():
        #for relevant data
        plt.scatter(x0[label.ne(2)], x1[label.ne(2)], c=label[label.ne(2)], s=500, cmap=plt.cm.get_cmap("jet",3), marker='.')
        #for distractor
        plt.scatter(x0[label.eq(2)], x1[label.eq(2)], c='green', s=200, marker='+')
        #ticks means that only [0,1,2] are used for label on colorbar. If ticks do not exist, float numbers appear on colorbar
    #######plt.colorbar(label='label', ticks=range(len(label.unique())))
    plt.xlim(0,1)
    plt.ylim(0,1)
    path = '/mnt_host/share_nfs/output_log/ATL_ouput/simple_test/'+file_name
    plt.savefig(path, bbox_inches='tight')
    plt.clf()


tr_data = []
te_data = []
''' Training data: binary classification(label: 0, 2)'''
ex_0 =  normal_sampling(0.1, 0.05, 0.9, 0.05, 10, 0)
ex_1 = normal_sampling(0.8, 0.05, 0.2, 0.05, 10, 1)
tr_data = [torch.cat([ex_0[0], ex_1[0]],0), torch.cat([ex_0[1], ex_1[1]],0), torch.cat([ex_0[2], ex_1[2]],0)]
draw_examples(tr_data[0], tr_data[1], tr_data[2], 'train_input_space')

''' Test data (unlabeled data): binary classification(label: 0, 2) distractor(label:1)'''
ex_0 = normal_sampling(0.2, 0.1, 0.8, 0.1, 10, 0)
ex_1 = normal_sampling(0.7, 0.1, 0.3, 0.1, 10, 1)
te_data = [torch.cat([ex_0[0], ex_1[0]],0), torch.cat([ex_0[1], ex_1[1]],0), torch.cat([ex_0[2], ex_1[2]],0)]
ex_2 = normal_sampling(0.7, 0.1, 0.8, 0.1, 10, 2)
te_data = [torch.cat([te_data[0], ex_2[0]],0), torch.cat([te_data[1], ex_2[1]],0), torch.cat([te_data[2], ex_2[2]],0)]
ex_2_2 = normal_sampling(0.6, 0.1, 0.4, 0.1, 10, 2)
te_data = [torch.cat([te_data[0], ex_2[0]],0), torch.cat([te_data[1], ex_2[1]],0), torch.cat([te_data[2], ex_2[2]],0)]
draw_examples(te_data[0], te_data[1], te_data[2], 'test_input_space')

#NOTE: If labels.shape==[2], mse_loss is failed since out.shape==[2,1]
#NOTE: If labels.shape==[2,1], mse_loss succeed since out.shape==[2,1]
#NOTE: In evaluation like discriminator, if reldata==[0.5,0.5] distractor==[-0.5,-0.5]. Relevant data distance and Distractor distance are the same (training with training dataset, test with training dataset)
#NOTE: In evaluation like image classifier, result is the same with above

model = torch.nn.Sequential(
    torch.nn.Linear(2,2, bias=True),
    #torch.nn.ReLU(),
    #torch.nn.Linear(2,2, bias=True),
)
op = torch.optim.SGD(model.parameters(), lr=0.01)
model.train()
op.zero_grad()

''''''
'''TRAIN SET (labeled data) train and draw'''
''''''
# exs[0] dim:[examples_num], exs[1] dim:[examples_num] -> [batch, examples_num]
inputs = torch.cat([torch.FloatTensor([[exs[0], exs[1]]]) for exs in \
                                        zip(tr_data[0], tr_data[1])], 0)
ex_2_3 = normal_sampling(0.6, 0.1, 0.4, 0.1, 10, 2)
inputs_distractor = torch.cat([torch.FloatTensor([[exs[0], exs[1]]]) for exs in \
                                        zip(ex_2_3[0], ex_2_3[1])], 0)
out = None
#NOTE: no distractor loss epochs 70
#NOTE: distractor loss epochs 80
for epoch in range(80):
    out = model(inputs)
    #NOTE: softmax(out, 1) -> (out, dim) -> ex) feature space[0.1, 0.4] -> softmax[0.1, 0.9]
    out = torch.nn.functional.softmax(out, 1)
    labels = tr_data[2]
    loss = torch.nn.functional.cross_entropy(out, labels)
    '''distractor loss'''
    out_dt = model(inputs_distractor)
    out_dt = torch.nn.functional.softmax(out, 1)
    label_max_prob = out_dt.max(1)[0]
    loss_distractor = torch.ones(len(out_dt))-label_max_prob
    loss = loss + loss_distractor.mean()
    '''end'''
    loss.backward()
    if epoch%10==0:print('loss', loss.detach().numpy(), end=' ')
    #feature space
    if epoch%10==0:print('inputs fs', out[0].detach().numpy())
    op.step()

out_x0 = torch.cat([x0.detach().view(1) for x0, x1 in out], 0)
out_x1 = torch.cat([x1.detach().view(1) for x0, x1 in out], 0)
draw_examples(out_x0, out_x1, tr_data[2], 'softmax_train')

''''''
'''TEST SET (unlabeled data) forward and draw'''
''''''
# exs[0] dim:[examples_num], exs[1] dim:[examples_num] -> [batch, examples_num]
te_inputs = torch.cat([torch.FloatTensor([[exs[0], exs[1]]]) for exs in \
                                        zip(te_data[0], te_data[1])], 0)
out = model(te_inputs)
out = torch.nn.functional.softmax(out, 1)
#NOTE: x0 dim:[] -> x0.view(1) dim:[1]. This is for torch.cat([...],0)
out_x0 = torch.cat([x0.detach().view(1) for x0, x1 in out], 0)
out_x1 = torch.cat([x1.detach().view(1) for x0, x1 in out], 0)
#NOTE: To adjust color of distractor with cmap in plt.scatter, we should exchange labels of 1:target label 2:distractor -> 2:target label 1:distractor because in softmax_train, there are only two labels and two colors(blue:0, brown:1) but in softmax_test, there are three labels[0,1,2] and three colors(blue:0, green:1, brown:2)
draw_examples(out_x0, out_x1, te_data[2], 'softmax_test')

print('w', list(model.parameters()))

'''
#Evaluation like Discriminator
###
for epoch in range(1000):
    if epoch%200==0:print('w', linear.weight[0][0].detach().numpy(), linear.weight[0][1].detach().numpy(), end=' ')
    out = linear(inputs)
    out = torch.nn.functional.sigmoid(out)
    loss = torch.nn.functional.mse_loss(out, labels)
    loss.backward()
    if epoch%200==0:print('loss', loss.detach().numpy(), end=' ')
    #feature space
    if epoch%200==0:print('inputs fs', out[0].detach().numpy(), out[1].detach().numpy())
    op.step()

with torch.no_grad():
    #feature space
    print('Relevant data fs', torch.nn.functional.sigmoid(linear(reldata)))
    print('Relevant data distance', torch.abs(boundary-torch.nn.functional.sigmoid(linear(reldata))))
    print('Distractor fs', torch.nn.functional.sigmoid(linear(distractor)))
    print('Distractor distance', torch.abs(boundary-torch.nn.functional.sigmoid(linear(distractor))))

print('w', linear.weight)
###

'''
