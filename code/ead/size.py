import torch

# 모델 정의
def get_pdn(out=384):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 256, 4), torch.nn.ReLU(inplace=True),
        torch.nn.AvgPool2d(2, 2),
        torch.nn.Conv2d(256, 512, 4), torch.nn.ReLU(inplace=True),
        torch.nn.AvgPool2d(2, 2),
        torch.nn.Conv2d(512, 512, 1), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(512, 512, 3), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(512, out, 4), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(out, out, 1)
    )

def get_ae():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 4, 2, 1), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(32, 32, 4, 2, 1), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(32, 64, 4, 2, 1), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 64, 4, 2, 1), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 64, 4, 2, 1), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 64, 8),
        torch.nn.Upsample(3, mode='bilinear'),
        torch.nn.Conv2d(64, 64, 4, 1, 2), torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(8, mode='bilinear'),
        torch.nn.Conv2d(64, 64, 4, 1, 2), torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(15, mode='bilinear'),
        torch.nn.Conv2d(64, 64, 4, 1, 2), torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(32, mode='bilinear'),
        torch.nn.Conv2d(64, 64, 4, 1, 2), torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(63, mode='bilinear'),
        torch.nn.Conv2d(64, 64, 4, 1, 2), torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(127, mode='bilinear'),
        torch.nn.Conv2d(64, 64, 4, 1, 2), torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(56, mode='bilinear'),
        torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 384, 3, 1, 1)
    )

# 모델 인스턴스 생성
teacher_model = get_pdn(384)
student_model = get_pdn(768)
autoencoder_model = get_ae()

# 파라미터 수 계산 함수
def model_param_count(model):
    return sum(p.numel() for p in model.parameters())

# 결과 출력
print("Teacher 모델 파라미터 수:", model_param_count(teacher_model))
print("Student 모델 파라미터 수:", model_param_count(student_model))
print("Autoencoder 모델 파라미터 수:", model_param_count(autoencoder_model))
