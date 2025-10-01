import torch

# # print("CUDA available:", torch.cuda.is_available())
# # print("GPU count:", torch.cuda.device_count())

# # for i in range(torch.cuda.device_count()):
# #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# import torch
# x = torch.rand(5000, 5000).to("cuda")
# for _ in range(1000):
#     x = x @ x
# weights = torch.load('C:\WoodAI\src\save_model\Go_v4_epoch10.pth')
# print(type(weights))
# state_dict = torch.load(r'C:\WoodAI\src\save_model\Go_v4_epoch10.pth')
# print(list(state_dict.keys())[:20])  # In 20 key đầu

