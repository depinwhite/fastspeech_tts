from src.config import TrainConfig, FastSpeechConfig, MelSpectrogramConfig
import torch.nn as nn
from src.modules import FastSpeech, FastSpeechLoss
from src.utils import get_data_to_buffer, reprocess_tensor
from src.dataset import BufferDataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from src.wandb_writer import WanDBWriter
import os

train_config = TrainConfig()

def collate_fn_tensor(batch):
    global train_config
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // train_config.batch_expand_size

    cut_list = list()
    for i in range(train_config.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(train_config.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output


def train_loop(training_loader, model):
    current_step = 0
    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)
    fastspeech_loss = FastSpeechLoss()
    
    logger = WanDBWriter(train_config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
        })

    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                current_step += 1
                tqdm_bar.update(1)

                logger.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_output, duration_predictor_output = model(character,
                                                              src_pos,
                                                              mel_pos=mel_pos,
                                                              mel_max_length=max_mel_len,
                                                              length_target=duration)

                # Calc Loss
                mel_loss, duration_loss = fastspeech_loss(mel_output,
                                                        duration_predictor_output,
                                                        mel_target,
                                                        duration)
                total_loss = mel_loss + duration_loss

                # Logger
                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                d_l = duration_loss.detach().cpu().numpy()

                logger.add_scalar("duration_loss", d_l)
                logger.add_scalar("mel_loss", m_l)
                logger.add_scalar("total_loss", t_l)

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    if not os.path.exists(train_config.checkpoint_path):
                        os.makedirs(train_config.checkpoint_path)

                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)


def main(model_config, mel_config):
    model = FastSpeech(model_config, mel_config)
    model = model.to(train_config.device)
    
    buffer = get_data_to_buffer(train_config)

    dataset = BufferDataset(buffer)

    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn_tensor,
        drop_last=True,
        num_workers=0
    )
    
    train_loop(training_loader, model)
    

if __name__ == "__main__":
    model_config = FastSpeechConfig()
    mel_config = MelSpectrogramConfig()
    main(model_config, mel_config)