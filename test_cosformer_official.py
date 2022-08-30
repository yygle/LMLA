import numpy as np
import torch
from torch.backends import cudnn
import tqdm
from test_ideas.test_cosformer.official_cosformer_attention_jit import CosformerAttention
cudnn.benchmark = True

device = 'cuda:5'
# model = resnet50().to(device)
embed_dim, num_heads, causal = 256, 4, False
model = CosformerAttention(embed_dim=embed_dim, num_heads=num_heads, causal=causal).to(device)
with open('./res_fast.txt', 'w', encoding='utf8') as fout:
    fout.write('embed_dim={}, num_heads={}, causal={}\n'.format(str(embed_dim), str(num_heads), str(causal)))
    fout.write('#'*20 + '\n')
    for tgt_len in range(100, 2001, 100):
        for fast_inf in [False, True]:
            batch = 100
            src_len = tgt_len
            repetitions = 1000
            query = torch.rand(tgt_len, batch, embed_dim).to(device)
            key = torch.rand(src_len, batch, embed_dim).to(device)
            value = torch.rand(src_len, batch, embed_dim).to(device)

            # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
            print('warm up ...\n')
            with torch.no_grad():
                for _ in range(100):
                    if not fast_inf:
                        left_res = model.left_product(query, key, value, None)
                    else:
                        right_res = model(query, key, value, None)

            # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
            torch.cuda.synchronize()


            # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # 初始化一个时间容器
            timings = np.zeros((repetitions, 1))

            print('testing ...\n')
            with torch.no_grad():
                for rep in tqdm.tqdm(range(repetitions)):
                    starter.record()
                    if not fast_inf:
                        left_res = model.left_product(query, key, value, None)
                    else:
                        right_res = model(query, key, value, None)
                    ender.record()
                    torch.cuda.synchronize() # 等待GPU任务完成
                    curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
                    timings[rep] = curr_time

            avg = timings.sum()/repetitions
            print('\nfast={},length={},avg={}\n'.format(fast_inf, tgt_len, avg))
            fout.write('fast={},length={},avg={}\n'.format(fast_inf, tgt_len, avg))

############ results ################

# embed_dim, num_heads, causal = 256, 4, False
# tgt_len, batch = 1000, 100
# left : 19.345100786924363
# right: 7.541637122631073
# tgt_len, batch = 500, 100
# left : 6.492434423446655
# right: 3.870778366804123
###############################
# tgt_len, batch = 1000, 100
# left :
# right:
###############################
