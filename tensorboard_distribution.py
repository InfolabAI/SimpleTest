import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for i in range(100):
    data = np.random.normal(0, 1, 50)
    writer.add_histogram('data_distribution1', data, global_step=i)
writer.close()
