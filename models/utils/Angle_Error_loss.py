import torch





class CosLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        # outputs, targets: (batch, frames, 3)

        l2 = torch.linalg.norm(outputs, ord=2, dim=-1, keepdim=True)  # (batch, frames, 1)
        outputs = outputs / (l2 + 1e-8)  # ゼロ除算防止
        l2 = torch.linalg.norm(targets, ord=2, dim=-1, keepdim=True)  # (batch, frames, 1)
        targets = targets / (l2 + 1e-8)

        cos =  torch.sum(outputs*targets, dim=-1)

        # 数値安定化
        cos = torch.clamp(cos, min=-0.999, max=0.999)

        rad = torch.acos(cos) # batch, frames
        #loss = torch.rad2deg(rad) # batch, frames
        loss = rad

        return loss.mean()

# cosine loss for single frame
class CosLossSingle(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        l2 = torch.linalg.norm(outputs, ord=2, axis=1)
        outputs = outputs/(l2[:,None]+ 1e-8)
        outputs = outputs.reshape(-1, outputs.shape[-1])

        l2 = torch.linalg.norm(targets, ord=2, axis=1)
        targets = targets/(l2[:,None]+ 1e-8)
        targets = targets.reshape(-1, targets.shape[-1])

        cos =  torch.sum(outputs*targets,dim=-1)
        cos[cos > 1] = 1
        cos[cos < -1] = -1
        rad = torch.acos(cos)
        #loss = torch.rad2deg(rad)#0.5*(1-cos)#criterion(pred_gaze,gaze_dir)
        loss = rad

        return loss

